#!/bin/bash

echo "Starting llama.cpp server..."

# Ensure the models directory exists
mkdir -p /app/models
MODEL_FILE="/app/models/Qwen3-Reranker-0.6B-q4_k_m.gguf"

# Check if model exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Model not found. Downloading Qwen3-Reranker-0.6B model..."
    wget -O "$MODEL_FILE" "https://huggingface.co/Mungert/Qwen3-Reranker-0.6B-GGUF/resolve/main/Qwen3-Reranker-0.6B-q4_k_m.gguf"
    
    if [ $? -eq 0 ]; then
        echo "Model downloaded successfully!"
    else
        echo "Failed to download model. Exiting."
        exit 1
    fi
else
    echo "Model already exists, skipping download."
fi

# Start the llama.cpp server with your exact parameters
exec /app/llama-server \
    --model "$MODEL_FILE" \
    --port 10000 \
    --host 0.0.0.0 \
    --ctx-size 4096 \
    --cont-batching