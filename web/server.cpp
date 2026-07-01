/**
 * @file server.cpp
 * @brief Web server for Vesper LLM inference
 * 
 * A simple HTTP server that loads a trained Vesper model and serves
 * text generation requests via REST API.
 * 
 * Usage: ./vesper_server [model_path] [tokenizer_path] [port]
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
#include <sentencepiece_processor.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <mutex>
#include <chrono>

using namespace vesper;
using json = nlohmann::json;

// Global model, tokenizer and mutex for thread-safe access
std::unique_ptr<models::TransformerLM> g_model;
std::unique_ptr<sentencepiece::SentencePieceProcessor> g_tokenizer;
std::mutex g_model_mutex;
Device g_device = Device::HIP;

// Model configuration (set when loading)
models::TransformerConfig g_config;
std::string g_model_path;
std::string g_tokenizer_path;

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

bool load_tokenizer(const std::string& tokenizer_path) {
    try {
        std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;
        
        g_tokenizer = std::make_unique<sentencepiece::SentencePieceProcessor>();
        auto status = g_tokenizer->Load(tokenizer_path);
        
        if (!status.ok()) {
            std::cerr << "Failed to load tokenizer: " << status.ToString() << std::endl;
            g_tokenizer.reset();
            return false;
        }
        
        std::cout << "Tokenizer loaded successfully!" << std::endl;
        std::cout << "  Vocab size: " << g_tokenizer->GetPieceSize() << std::endl;
        
        g_tokenizer_path = tokenizer_path;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load tokenizer: " << e.what() << std::endl;
        return false;
    }
}

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
    float temperature = 0.7f;   // Lower default for more coherent output
    int top_k = 50;             // Slightly wider top-k
};

std::string generate_text(const GenerationParams& params) {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    
    if (!g_model) {
        throw std::runtime_error("Model not loaded");
    }
    
    if (!g_tokenizer) {
        throw std::runtime_error("Tokenizer not loaded");
    }
    
    autograd::NoGradGuard no_grad;
    
    // Initialize cache for generation
    g_model->clear_cache();
    g_model->init_cache(1, g_device);
    
    // Tokenize the prompt using SentencePiece
    std::vector<int> prompt_tokens_int;
    g_tokenizer->Encode(params.prompt, &prompt_tokens_int);
    
    // Convert to int32_t
    std::vector<int32_t> prompt_tokens(prompt_tokens_int.begin(), prompt_tokens_int.end());
    
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(g_tokenizer->bos_id());  // BOS token
    }
    
    // Create tensor
    Tensor tokens = vesper::empty({1, static_cast<int64_t>(prompt_tokens.size())}, 
                                   DType::Int32, Device::CPU, false);
    tokens.copy_from_host(prompt_tokens.data());
    tokens = tokens.to(g_device);
    
    // Generate using the model's generate method
    Tensor output = g_model->generate(tokens, params.max_tokens, 
                                       params.temperature, params.top_k);
    
    // Convert output tokens back to string using SentencePiece
    Tensor output_cpu = output.to(Device::CPU);
    int64_t seq_len = output_cpu.shape()[1];  // [batch, seq_len]
    
    const int32_t* out_data = output_cpu.data_ptr<int32_t>();
    
    // Get only the generated tokens (after prompt), stopping at EOS
    int eos_id = g_tokenizer->eos_id();
    int vocab_size = g_tokenizer->GetPieceSize();
    std::vector<int> generated_tokens;
    std::cerr << "Generated token IDs: ";
    for (int64_t i = prompt_tokens.size(); i < seq_len; ++i) {
        int tok = static_cast<int>(out_data[i]);
        std::cerr << tok << " ";
        if (tok == eos_id) {
            break;  // Stop at EOS token
        }
        // Check for out-of-vocabulary tokens
        if (tok < 0 || tok >= vocab_size) {
            std::cerr << "(OOV!) ";
        }
        generated_tokens.push_back(tok);
    }
    std::cerr << std::endl;
    
    // Decode tokens back to text
    std::string result;
    g_tokenizer->Decode(generated_tokens, &result);
    
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
    bool ready = g_model && g_tokenizer;
    json response = {
        {"status", ready ? "healthy" : "not_ready"},
        {"model_loaded", g_model != nullptr},
        {"tokenizer_loaded", g_tokenizer != nullptr},
        {"model_path", g_model_path},
        {"tokenizer_path", g_tokenizer_path}
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
    std::string tokenizer_path = "../data/tinystories/tokenizer.model";
    int port = 8080;
    
    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        tokenizer_path = argv[2];
    }
    if (argc > 3) {
        port = std::stoi(argv[3]);
    }
    
    std::cout << "Usage: ./vesper_server [model_path] [tokenizer_path] [port]" << std::endl;
    std::cout << "  model_path:     " << model_path << std::endl;
    std::cout << "  tokenizer_path: " << tokenizer_path << std::endl;
    std::cout << "  port:           " << port << std::endl;
    std::cout << std::endl;
    
    // Model configuration (should match training config)
    // TODO: Save config alongside model or embed in checkpoint
    models::TransformerConfig config;
    config.vocab_size = 32000;
    config.dim = 1024;          // Match train_tinystories.cpp
    config.n_layers = 8;        // Match train_tinystories.cpp
    config.n_heads = 8;         // Match train_tinystories.cpp
    config.max_seq_len = 1024;  // Match training
    config.use_rms_norm = true;
    config.rope_base = 10000.0f;
    
    // Load tokenizer
    if (!load_tokenizer(tokenizer_path)) {
        std::cerr << "Warning: Starting server without tokenizer loaded." << std::endl;
    }
    
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
