/**
 * Vesper LLM Web UI - Application JavaScript
 * 
 * Handles communication with the Vesper server and UI interactions.
 */

// ===========================================================================
// Configuration
// ===========================================================================

const API_BASE = '';  // Same origin, or set to 'http://localhost:8080' for dev

// ===========================================================================
// State
// ===========================================================================

let isGenerating = false;
let modelConfig = null;

// Settings state
const settings = {
    maxTokens: 150,
    temperature: 0.8,
    topK: 40
};

// ===========================================================================
// DOM Elements
// ===========================================================================

const elements = {
    chatMessages: document.getElementById('chat-messages'),
    promptInput: document.getElementById('prompt-input'),
    sendBtn: document.getElementById('send-btn'),
    settingsBtn: document.getElementById('settings-btn'),
    settingsPanel: document.getElementById('settings-panel'),
    status: document.getElementById('status'),
    statusText: document.querySelector('.status-text'),
    
    // Settings
    maxTokensSlider: document.getElementById('max-tokens'),
    maxTokensValue: document.getElementById('max-tokens-value'),
    temperatureSlider: document.getElementById('temperature'),
    temperatureValue: document.getElementById('temperature-value'),
    topKSlider: document.getElementById('top-k'),
    topKValue: document.getElementById('top-k-value'),
    
    // Model info
    infoParams: document.getElementById('info-params'),
    infoDim: document.getElementById('info-dim'),
    infoLayers: document.getElementById('info-layers'),
    infoHeads: document.getElementById('info-heads'),
    infoSeq: document.getElementById('info-seq'),
    infoVocab: document.getElementById('info-vocab')
};

// ===========================================================================
// API Functions
// ===========================================================================

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Health check failed:', error);
        return { status: 'error', model_loaded: false };
    }
}

async function getModelConfig() {
    try {
        const response = await fetch(`${API_BASE}/api/config`);
        if (!response.ok) {
            throw new Error('Config endpoint not available');
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Failed to get model config:', error);
        return null;
    }
}

async function generateText(prompt) {
    const response = await fetch(`${API_BASE}/api/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: prompt,
            max_tokens: settings.maxTokens,
            temperature: settings.temperature,
            top_k: settings.topK
        })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Generation failed');
    }
    
    return await response.json();
}

// ===========================================================================
// UI Functions
// ===========================================================================

function updateStatus(status, message) {
    elements.status.className = 'status ' + status;
    elements.statusText.textContent = message;
}

function updateModelInfo(config) {
    if (!config) {
        elements.infoParams.textContent = '-';
        elements.infoDim.textContent = '-';
        elements.infoLayers.textContent = '-';
        elements.infoHeads.textContent = '-';
        elements.infoSeq.textContent = '-';
        elements.infoVocab.textContent = '-';
        return;
    }
    
    // Format parameters (e.g., "110M")
    const params = config.parameters;
    let paramStr = params.toLocaleString();
    if (params >= 1e9) {
        paramStr = (params / 1e9).toFixed(1) + 'B';
    } else if (params >= 1e6) {
        paramStr = (params / 1e6).toFixed(1) + 'M';
    } else if (params >= 1e3) {
        paramStr = (params / 1e3).toFixed(1) + 'K';
    }
    
    elements.infoParams.textContent = paramStr;
    elements.infoDim.textContent = config.dim;
    elements.infoLayers.textContent = config.n_layers;
    elements.infoHeads.textContent = config.n_heads;
    elements.infoSeq.textContent = config.max_seq_len;
    elements.infoVocab.textContent = config.vocab_size.toLocaleString();
}

function addMessage(content, type, meta = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Handle text with potential newlines
    if (typeof content === 'string') {
        content.split('\n').forEach((line, i, arr) => {
            contentDiv.appendChild(document.createTextNode(line));
            if (i < arr.length - 1) {
                contentDiv.appendChild(document.createElement('br'));
            }
        });
    } else {
        contentDiv.appendChild(content);
    }
    
    messageDiv.appendChild(contentDiv);
    
    if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        metaDiv.textContent = meta;
        messageDiv.appendChild(metaDiv);
    }
    
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

function addLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant-message loading-message';
    loadingDiv.innerHTML = `
        <div class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    elements.chatMessages.appendChild(loadingDiv);
    scrollToBottom();
    return loadingDiv;
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function setGenerating(generating) {
    isGenerating = generating;
    elements.sendBtn.disabled = generating;
    elements.promptInput.disabled = generating;
}

// ===========================================================================
// Event Handlers
// ===========================================================================

async function handleSend() {
    const prompt = elements.promptInput.value.trim();
    if (!prompt || isGenerating) return;
    
    // Clear input
    elements.promptInput.value = '';
    autoResizeTextarea();
    
    // Add user message
    addMessage(prompt, 'user');
    
    // Add loading indicator
    const loadingMsg = addLoadingMessage();
    
    setGenerating(true);
    
    try {
        const result = await generateText(prompt);
        
        // Remove loading indicator
        loadingMsg.remove();
        
        // Add assistant response
        const meta = `${result.tokens_generated} tokens · ${result.generation_time_ms}ms`;
        addMessage(result.full_text, 'assistant', meta);
        
    } catch (error) {
        // Remove loading indicator
        loadingMsg.remove();
        
        // Add error message
        addMessage(`Error: ${error.message}`, 'system');
        console.error('Generation error:', error);
    }
    
    setGenerating(false);
    elements.promptInput.focus();
}

function handleSettingsToggle() {
    elements.settingsPanel.classList.toggle('hidden');
}

function handleSettingChange(setting, value, displayElement) {
    settings[setting] = parseFloat(value);
    displayElement.textContent = value;
}

function autoResizeTextarea() {
    const textarea = elements.promptInput;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

// ===========================================================================
// Initialization
// ===========================================================================

async function init() {
    // Set up event listeners
    elements.sendBtn.addEventListener('click', handleSend);
    
    elements.promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
    
    elements.promptInput.addEventListener('input', autoResizeTextarea);
    
    elements.settingsBtn.addEventListener('click', handleSettingsToggle);
    
    // Settings sliders
    elements.maxTokensSlider.addEventListener('input', (e) => {
        handleSettingChange('maxTokens', e.target.value, elements.maxTokensValue);
    });
    
    elements.temperatureSlider.addEventListener('input', (e) => {
        handleSettingChange('temperature', e.target.value, elements.temperatureValue);
    });
    
    elements.topKSlider.addEventListener('input', (e) => {
        handleSettingChange('topK', e.target.value, elements.topKValue);
    });
    
    // Initial health check
    updateStatus('', 'Connecting...');
    
    const health = await checkHealth();
    
    if (health.status === 'healthy' && health.model_loaded) {
        updateStatus('connected', 'Model loaded');
        
        // Get model config
        modelConfig = await getModelConfig();
        updateModelInfo(modelConfig);
        
    } else if (health.status === 'no_model') {
        updateStatus('warning', 'No model loaded');
        updateModelInfo(null);
        
    } else {
        updateStatus('error', 'Server unavailable');
        updateModelInfo(null);
    }
    
    // Focus input
    elements.promptInput.focus();
    
    // Periodic health checks
    setInterval(async () => {
        const health = await checkHealth();
        if (health.status === 'healthy' && health.model_loaded) {
            updateStatus('connected', 'Model loaded');
        } else if (health.status === 'no_model') {
            updateStatus('warning', 'No model loaded');
        } else {
            updateStatus('error', 'Connection lost');
        }
    }, 30000);
}

// Start the app
document.addEventListener('DOMContentLoaded', init);
