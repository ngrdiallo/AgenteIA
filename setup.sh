#!/bin/bash
# IAGestioneArte - Setup script
# Usage: ./setup.sh

echo "============================================"
echo "  IAGestioneArte - Configurazione"
echo "============================================"
echo ""

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  Il file .env esiste già."
    read -p "Vuoi sovrascriverlo? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "Setup annullato."
        exit 0
    fi
fi

# Create .env from template
echo ""
echo "Creazione file .env..."

cat > .env << 'EOF'
# ============================================================
# IAGestioneArte — API Keys Configuration
# ============================================================

# PRIMARY LLM BACKEND
OPENROUTER_API_KEY=your_openrouter_key_here

# Google Gemini (per vision + LLM fallback)
GOOGLE_API_KEY=your_google_api_key_here

# Groq (ultra-fast fallback)
GROQ_API_KEY=your_groq_api_key_here

# Mistral (reliable fallback)
MISTRAL_API_KEY=your_mistral_api_key_here

# HuggingFace (inference API)
HF_TOKEN=your_huggingface_token_here

# DeepSeek (vision analysis)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Cohere (opzionale — reranker cloud)
COHERE_API_KEY=your_cohere_api_key_here

# Cerebras (ultra-fast 1800 tok/s — free da cloud.cerebras.ai)
CEREBRAS_API_KEY=your_cerebras_api_key_here

# SambaNova (Llama 405B gratis — free da cloud.sambanova.ai)
SAMBANOVA_API_KEY=your_sambanova_api_key_here

# === Chiavi aggiunte per espansione provider ===
OPENAI_API_KEY=your_openai_api_key_here

XAI_API_KEY=your_xai_api_key_here

NVIDIA_API_KEY=your_nvidia_api_key_here

HYPERBOLIC_API_KEY=your_hyperbolic_api_key_here

FIREWORKS_API_KEY=your_fireworks_api_key_here

CHUTES_API_KEY=your_chutes_api_key_here

TAVILY_API_KEY=your_tavily_api_key_here

SERPAPI_API_KEY=your_serpapi_api_key_here

STABILITY_API_KEY=your_stability_api_key_here

# IO.NET Intelligence (500K token/giorno gratis)
# Registrati su https://intelligence.io.solutions/
IONET_API_KEY=your_ionet_api_key_here

# ModelScope
MODELSCOPE_API_KEY=your_modelscope_api_key_here

EOF

echo "✅ File .env creato!"
echo ""
echo "============================================"
echo "  Prossimi passi:"
echo "============================================"
echo ""
echo "1. Modifica il file .env con le tue API key"
echo "   (le chiavi attuali sono placeholder)"
echo ""
echo "2. Per ottenere le API key gratis:"
echo "   - Groq: https://console.groq.com/"
echo "   - Cerebras: https://cloud.cerebras.ai/"
echo "   - IO.NET: https://intelligence.io.solutions/"
echo "   - OpenRouter: https://openrouter.ai/"
echo "   - Fireworks: https://fireworks.ai/"
echo ""
echo "3. Avvia il server:"
echo "   ./run.sh"
echo ""
