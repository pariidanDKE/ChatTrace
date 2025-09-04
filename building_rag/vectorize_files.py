import pandas as pd
import time
import os
from building_rag.custom_textsplitter import CustomTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")



def split_messages(df,chunk_size=1000):
    start_time = time.time()
    splitter = CustomTextSplitter()
    documents = splitter.split_messages(chunk_size=chunk_size, message_df=df)
    print(f"⏱️  Split messages: {time.time() - start_time:.2f}s")
    return documents

def embed_documents(documents,source):

    embedding_model_name = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
    start_time = time.time()
    # Initialize embedding model
    embeddings = OllamaEmbeddings(
        model=embedding_model_name,
        temperature=0,
        base_url = OLLAMA_BASE_URL,
    )
    vector_store = Chroma(
        collection_name=f"chat_documents_{source}_{embedding_model_name.replace(":",'-').replace('/','-')}",
        embedding_function=embeddings,
        persist_directory='.chroma_db-test'
    )
    
    doc_names = [(doc.metadata['chat_name'] + '(' + str(doc.metadata['date_range']) + '_' + str(doc.metadata['part']) + ')' + '_' + str(doc.metadata['chat_id'])) for doc in documents]
    vector_store.add_documents(documents, ids=doc_names)
    print(f"⏱️  Embed documents: {time.time() - start_time:.2f}s")

def vectorize_chats(chat_path = 'test_data/processed_chats/whatsapp_chats.csv',source = 'whatsapp', chunk_size = 2000):
    total_start = time.time()
    # load data
    load_start = time.time()
    message_df = pd.read_csv(chat_path)

    message_df = message_df.sample(500)
    print(f"⏱️  Load data: {time.time() - load_start:.2f}s")
    
    # split messages
    documents = split_messages(message_df,chunk_size=chunk_size)
    
    # embed documents
    embed_documents(documents,source)
    print(f"⏱️  Total time: {time.time() - total_start:.2f}s")

if __name__ == '__main__':
    vectorize_chats()