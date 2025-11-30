/**
 * @file server.cpp
 * @brief Web server for Vesper LLM inference
 * 
 * A simple HTTP server that loads a trained Vesper model and serves
 * text generation requests via REST API.
 * 
 * Usage: ./vesper_server [model_path] [port]
 * 
 * Endpoints:
 *   GET  /              - Serves the web UI
 *   GET  /health        - Health check
 *   POST /api/generate  - Generate text from prompt
 *   GET  /api/config    - Get model configuration
 */

#define CPPHTTPLIB_THREAD_POOL_COUNT 4
#include "httplib.h"
#include "json.hpp"

#include <vesper/vesper.h>
#include <vesper/serialization.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <mutex>
#include <chrono>

using namespace vesper;
using json = nlohmann::json;

// Global model and mutex for thread-safe access
std::unique_ptr<models::TransformerLM> g_model;
std::mutex g_model_mutex;
Device g_device = Device::HIP;

// Model configuration (set when loading)
models::TransformerConfig g_config;
std::string g_model_path;

// =============================================================================
// Utility Functions
// =============================================================================

std::string read_file_contents(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string get_content_type(const std::string& path) {
    // C++17 compatible string suffix checking
    auto ends_with = [](const std::string& str, const std::string& suffix) {
        if (suffix.size() > str.size()) return false;
        return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    
    if (ends_with(path, ".html")) return "text/html";
    if (ends_with(path, ".css")) return "text/css";
    if (ends_with(path, ".js")) return "application/javascript";
    if (ends_with(path, ".json")) return "application/json";
    if (ends_with(path, ".png")) return "image/png";
    if (ends_with(path, ".ico")) return "image/x-icon";
    return "text/plain";
}

// =============================================================================
// Model Loading
// =============================================================================

bool load_model(const std::string& model_path, const models::TransformerConfig& config) {
    try {
        std::cout << "Loading model from: " << model_path << std::endl;
        
        // Create model with config
        g_model = models::create_model(config);
        g_model->to(g_device);
        
        // Load weights
        vesper::load(*g_model, model_path);
        
        // Set to eval mode
        g_model->eval();
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "  Parameters: " << g_model->num_parameters() << std::endl;
        std::cout << "  Config: " << config.describe() << std::endl;
        
        g_config = config;
        g_model_path = model_path;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

// =============================================================================
// Text Generation
// =============================================================================

struct GenerationParams {
    std::string prompt;
    int max_tokens = 100;
    float temperature = 0.8f;
    int top_k = 40;
};

std::string generate_text(const GenerationParams& params) {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    
    if (!g_model) {
        throw std::runtime_error("Model not loaded");
    }
    
    autograd::NoGradGuard no_grad;
    
    // For now, create a simple token sequence from the prompt
    // In production, you'd use a proper tokenizer
    // This assumes the model was trained with a tokenizer that's compatible
    
    // Initialize cache for generation
    g_model->clear_cache();
    g_model->init_cache(1, g_device);
    
    // Create initial token tensor from prompt
    // Note: This is a placeholder - you'd need proper tokenization
    std::vector<int32_t> prompt_tokens;
    
    // Simple ASCII tokenization (for demo purposes)
    // In production, use the same tokenizer as training
    for (char c : params.prompt) {
        prompt_tokens.push_back(static_cast<int32_t>(c) % g_config.vocab_size);
    }
    
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(0);  // BOS token
    }
    
    // Create tensor
    Tensor tokens = vesper::empty({1, static_cast<int64_t>(prompt_tokens.size())}, 
                                   DType::Int32, Device::CPU, false);
    tokens.copy_from_host(prompt_tokens.data());
    tokens = tokens.to(g_device);
    
    // Generate using the model's generate method
    Tensor output = g_model->generate(tokens, params.max_tokens, 
                                       params.temperature, params.top_k);
    
    // Convert output tokens back to string
    Tensor output_cpu = output.to(Device::CPU);
    int64_t seq_len = output_cpu.shape()[1];  // [batch, seq_len]
    std::string result;
    
    const int32_t* out_data = output_cpu.data_ptr<int32_t>();
    for (int64_t i = prompt_tokens.size(); i < seq_len; ++i) {
        int32_t tok = out_data[i];
        // Simple ASCII decoding (for demo)
        if (tok >= 32 && tok < 127) {
            result += static_cast<char>(tok);
        }
    }
    
    g_model->clear_cache();
    
    return result;
}

// =============================================================================
// HTTP Handlers
// =============================================================================

void handle_generate(const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = json::parse(req.body);
        
        GenerationParams params;
        params.prompt = body.value("prompt", "");
        params.max_tokens = body.value("max_tokens", 100);
        params.temperature = body.value("temperature", 0.8f);
        params.top_k = body.value("top_k", 40);
        
        if (params.prompt.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "prompt is required"}}.dump(), "application/json");
            return;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string generated = generate_text(params);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        json response = {
            {"generated_text", generated},
            {"prompt", params.prompt},
            {"full_text", params.prompt + generated},
            {"tokens_generated", static_cast<int>(generated.length())},
            {"generation_time_ms", duration_ms}
        };
        
        res.set_content(response.dump(), "application/json");
        
    } catch (const json::parse_error& e) {
        res.status = 400;
        res.set_content(json{{"error", "Invalid JSON"}}.dump(), "application/json");
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

void handle_health(const httplib::Request& req, httplib::Response& res) {
    json response = {
        {"status", g_model ? "healthy" : "no_model"},
        {"model_loaded", g_model != nullptr},
        {"model_path", g_model_path}
    };
    res.set_content(response.dump(), "application/json");
}

void handle_config(const httplib::Request& req, httplib::Response& res) {
    if (!g_model) {
        res.status = 503;
        res.set_content(json{{"error", "Model not loaded"}}.dump(), "application/json");
        return;
    }
    
    json response = {
        {"vocab_size", g_config.vocab_size},
        {"dim", g_config.dim},
        {"n_layers", g_config.n_layers},
        {"n_heads", g_config.n_heads},
        {"max_seq_len", g_config.max_seq_len},
        {"parameters", g_model->num_parameters()},
        {"model_path", g_model_path}
    };
    res.set_content(response.dump(), "application/json");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Vesper LLM Web Server" << std::endl;
    std::cout << "============================================\n" << std::endl;
    
    // Parse arguments
    std::string model_path = "../checkpoints/model_final.bin";
    int port = 8080;
    
    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        port = std::stoi(argv[2]);
    }
    
    // Model configuration (should match training config)
    // TODO: Save config alongside model or embed in checkpoint
    models::TransformerConfig config;
    config.vocab_size = 32000;
    config.dim = 768;
    config.n_layers = 12;
    config.n_heads = 12;
    config.max_seq_len = 512;
    config.use_rms_norm = true;
    config.rope_base = 10000.0f;
    
    // Load model
    if (!load_model(model_path, config)) {
        std::cerr << "Warning: Starting server without model loaded." << std::endl;
        std::cerr << "Use /api/load endpoint to load a model." << std::endl;
    }
    
    // Create HTTP server
    httplib::Server svr;
    
    // Set up CORS for development
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"}
    });
    
    // Handle OPTIONS preflight requests
    svr.Options(".*", [](const httplib::Request& req, httplib::Response& res) {
        res.status = 204;
    });
    
    // API endpoints
    svr.Post("/api/generate", handle_generate);
    svr.Get("/api/health", handle_health);
    svr.Get("/api/config", handle_config);
    
    // Serve static files from web/ directory
    // When running from build/web/, static files are in ../../web/
    svr.set_mount_point("/", "../../web");
    
    // Also try current directory (for when running from web/)
    svr.set_mount_point("/static", ".");
    
    // Serve index.html for root
    svr.Get("/", [](const httplib::Request& req, httplib::Response& res) {
        try {
            // Try multiple paths
            std::vector<std::string> paths = {
                "../../web/index.html",
                "../web/index.html", 
                "index.html",
                "web/index.html"
            };
            
            for (const auto& path : paths) {
                std::ifstream file(path);
                if (file.is_open()) {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    res.set_content(buffer.str(), "text/html");
                    return;
                }
            }
            
            res.status = 404;
            res.set_content("index.html not found", "text/plain");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
        }
    });
    
    // Serve static CSS
    svr.Get("/style.css", [](const httplib::Request& req, httplib::Response& res) {
        std::vector<std::string> paths = {"../../web/style.css", "../web/style.css", "style.css"};
        for (const auto& path : paths) {
            std::ifstream file(path);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                res.set_content(buffer.str(), "text/css");
                return;
            }
        }
        res.status = 404;
    });
    
    // Serve static JS
    svr.Get("/app.js", [](const httplib::Request& req, httplib::Response& res) {
        std::vector<std::string> paths = {"../../web/app.js", "../web/app.js", "app.js"};
        for (const auto& path : paths) {
            std::ifstream file(path);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                res.set_content(buffer.str(), "application/javascript");
                return;
            }
        }
        res.status = 404;
    });
    
    std::cout << "Starting server on http://localhost:" << port << std::endl;
    std::cout << "Press Ctrl+C to stop\n" << std::endl;
    
    if (!svr.listen("0.0.0.0", port)) {
        std::cerr << "Failed to start server on port " << port << std::endl;
        return 1;
    }
    
    return 0;
}
