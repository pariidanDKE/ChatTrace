import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from reranker import Reranker

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

vector_store = None
chat_model = None
rerank_model = None

def load_chroma_db(model_name,temperature,collection_name,context_len):
    print(f"Setting base url : {OLLAMA_BASE_URL}")
    embeddings = OllamaEmbeddings(
        model=model_name,
        temperature=temperature,
        num_ctx = context_len,
        base_url = OLLAMA_BASE_URL
    )
    collection_name = collection_name + '_' + model_name.replace(':','-')

    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name="chat_documents_whatsapp_Qwen3-Embedding-0.6B-Q8_0-latest",#collection_name,
        persist_directory='.chroma_db',
    )
    return vector_store


### Oberservation : I did not want to increase context size becasue
def get_vector_store(model_name="Qwen3-Embedding-0.6B-Q8_0:latest",temperature=0,collection_name="chat_documents_whatsapp_ck2000", context_len=4096): #  chat_documents chat_documents_whatsapp_ck2000
    global vector_store
    if vector_store:
        return vector_store
    else:
        vector_store = load_chroma_db(model_name,temperature,collection_name,context_len)
        return vector_store

def get_chat_model(model_name="qwen2.5:7b-instruct", max_new_tokens=256, context_len=4096):
    global chat_model
    if chat_model:
        return chat_model
    else:
        chat_model = ChatOllama(
        base_url = OLLAMA_BASE_URL,
        model=model_name,
        num_predict=max_new_tokens,
        num_ctx=context_len,
        system="You are a chat analysis agent that retrieves and analyzes conversation data to answer user questions about their chat history.",
        reasoning = False
        )

        return chat_model
    

def get_rerank_model(model_name="Qwen/Qwen3-Reranker-0.6B", max_length=256):
    global rerank_model
    if rerank_model:
        return rerank_model
    else:
        rerank_model = Reranker(model_name,llama_cpp=True)
        return rerank_model


def format_response_metadata(step_message):
    """Format response metadata into a readable summary for verbose output."""
    
    def fmt_dur(ns): 
        if not ns: return "N/A"
        if ns >= 1e9: return f"{ns/1e9:.2f}s"
        if ns >= 6e10: return f"{ns/6e10:.1f}m"
        return f"{ns/1e6:.1f}ms"
    
    def fmt_speed(tokens, dur_ns): 
        return f"{tokens/(dur_ns/1e9):.1f} tok/s" if tokens and dur_ns else "N/A"
    
    usage = step_message.usage_metadata
    meta = step_message.response_metadata
    
    # Basic info
    model = meta.get('model', 'unknown')
    status = "✓" if meta.get('done') else "⏳"
    reason = meta.get('done_reason', 'unknown')
    created = meta.get('created_at', 'N/A')
    
    # Timing
    total_time = fmt_dur(meta.get('total_duration'))
    load_time = fmt_dur(meta.get('load_duration'))
    
    # Token counts
    in_tokens = usage.get('input_tokens') or meta.get('prompt_eval_count', 0)
    out_tokens = usage.get('output_tokens') or meta.get('eval_count', 0)
    total_tokens = usage.get('total_tokens', in_tokens + out_tokens)
    
    # Processing times and speeds
    prompt_time = fmt_dur(meta.get('prompt_eval_duration'))
    prompt_speed = fmt_speed(in_tokens, meta.get('prompt_eval_duration'))
    
    output_time = fmt_dur(meta.get('eval_duration'))
    output_speed = fmt_speed(out_tokens, meta.get('eval_duration'))
    
    return f"""Model: {model} {status}
Status: {reason}
Created: {created}
Total Time: {total_time} (Load: {load_time})
Tokens: {total_tokens} total
  Input: {in_tokens} tokens in {prompt_time} ({prompt_speed})
  Output: {out_tokens} tokens in {output_time} ({output_speed})"""
