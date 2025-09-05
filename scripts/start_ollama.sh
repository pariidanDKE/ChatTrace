#!/bin/bash
echo "Starting Ollama server ..."
ollama serve &
sleep 5

#MODEL_NAME="qwen3:8b"
# Check if model exists
if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "Pulling $MODEL_NAME model..."
    ollama pull "$MODEL_NAME"
else
    echo "Model $MODEL_NAME already exists, skipping pull."
fi

#EMBD_MODEL_NAME="dengcao/Qwen3-Embedding-0.6B:Q8_0"
# Check if model exists
if ! ollama list | grep -q "$EMBD_MODEL_NAME"; then
    echo "Pulling $EMBD_MODEL_NAME model..."
    ollama pull "$EMBD_MODEL_NAME"
else
    echo "Embedding Model $EMBD_MODEL_NAME already exists, skipping pull."
fi

wait

