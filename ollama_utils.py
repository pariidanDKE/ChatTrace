from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

vector_store = None
chat_model = None

def load_chroma_db(model_name,temperature,collection_name):
    embeddings = OllamaEmbeddings(
        model=model_name,
        temperature=temperature,
    )
    collection_name = collection_name + '_' + model_name.replace(':','-')


    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory='.chroma_db',
    )
    return vector_store

def get_vector_store(model_name="Qwen3-Embedding-0.6B-Q8_0:latest",temperature=0,collection_name="chat_documents_instagram"): # 
    global vector_store
    if vector_store:
        return vector_store
    else:
        vector_store = load_chroma_db(model_name,temperature,collection_name)
        return vector_store

def get_chat_model(model_name="qwen2.5:7b-instruct", max_new_tokens=256, context_len=4096):
    global chat_model
    if chat_model:
        return chat_model
    else:
        chat_model = ChatOllama(
        model=model_name,
        num_predict=max_new_tokens,
        num_ctx=context_len,
        system="You are a WhatsApp chat analysis agent that retrieves and analyzes conversation data to answer user questions about their chat history.",
        reasoning = False
        )

        return chat_model