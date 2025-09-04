#!/bin/bash
echo "Starting Ollama server ..."
ollama serve &
sleep 5

MODEL_NAME="qwen3:8b"

# Check if model exists
if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "Pulling $MODEL_NAME model..."
    ollama pull "$MODEL_NAME"
else
    echo "Model $MODEL_NAME already exists, skipping pull."
fi

#embedding model test:
EMBD_MODEL_NAME="dengcao/Qwen3-Embedding-0.6B:Q8_0"
# Check if model exists
if ! ollama list | grep -q "$EMBD_MODEL_NAME"; then
    echo "Pulling $EMBD_MODEL_NAME model..."
    ollama pull "$EMBD_MODEL_NAME"
else
    echo "Embedding Model $EMBD_MODEL_NAME already exists, skipping pull."
fi

# Check if embedding model exists
# if ! ollama list | grep -iq "embed"; then
#     echo "No embedding model found. Creating custom embedding model..."
#     GGUF_FILE="/app/scripts/Qwen3-Embedding-0.6B-GGUF.gguf"
#     if [ ! -f "$GGUF_FILE" ]; then
#         echo "Downloading GGUF model from Hugging Face..."
#         wget -O "$GGUF_FILE" "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf"
                              
#     fi
#     ollama create Qwen3-Embedding-0.6B-Q8_0 -f scripts/qwen3_embd_modelfile
# else
#     echo "Embedding model already exists, skipping creation."
# fi

# server is now background process, so should not let this scripts terminate as it will termiante all other processes
wait

