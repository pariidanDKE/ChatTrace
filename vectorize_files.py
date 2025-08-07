import pandas as pd
import time
from custom_textsplitter import CustomTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def split_messages(df):
    start_time = time.time()
    splitter = CustomTextSplitter()
    documents = splitter.split_messages(chunk_size=1000, message_df=df)
    print(f"⏱️  Split messages: {time.time() - start_time:.2f}s")
    return documents

def embed_documents(documents):
    start_time = time.time()
    # Initialize embedding model
    embeddings = OllamaEmbeddings(
        model="Qwen3-Embedding-0.6B-Q8_0:latest",
        temperature=0,
    )

    vector_store = Chroma(
        collection_name="chat_documents",
        embedding_function=embeddings,
        persist_directory='.chroma_db'
    )
    
    # import random
    # sample_documents = random.sample(documents, 100)
    doc_names = [(doc.metadata['chat_name'] + '(' + str(doc.metadata['date_range']) + '_' + str(doc.metadata['part']) + ')') for doc in documents]
    vector_store.add_documents(documents, ids=doc_names)
    print(f"⏱️  Embed documents: {time.time() - start_time:.2f}s")

def main():
    total_start = time.time()
    # load data
    load_start = time.time()
    message_df = pd.read_csv('/Users/pariidan/Documents/python_projects/RAG_Langchain/data/processed_data/whatsapp_chats.csv', index_col=0)
    print(f"⏱️  Load data: {time.time() - load_start:.2f}s")
    
    # split messages
    documents = split_messages(message_df)
    
    # embed documents
    embed_documents(documents)
    print(f"⏱️  Total time: {time.time() - total_start:.2f}s")

if __name__ == '__main__':
    main()