
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import glob
import hashlib

import config

# Retriever configuration
RETRIEVER_CONFIG = {
    "search_type": "mmr",
    "search_kwargs": {"k": 10}
}

def generate_patient_data_checksum(patient_id: str) -> str:
    """Generate single checksum for all patient data files to detect if something has changed."""
    hasher = hashlib.sha256()
    markdown_files = sorted(glob.glob(f"data/{patient_id}/**/*.md", recursive=True))
    
    for file_path in markdown_files:
        hasher.update(file_path.encode())
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
    
    return hasher.hexdigest()

def load_documents(patient_id: str):
    """Load and split patient documents."""
    documents = []
    markdown_files = glob.glob(f"data/{patient_id}/**/*.md", recursive=True)
    
    for file_path in markdown_files:
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250, separators=["\n---\n", "\n# ", "\n## ", "\n"])
    return text_splitter.split_documents(documents)

def create_retriever(patient_id) -> VectorStoreRetriever:
    """Load medical documents, create embeddings, store in a vector store, and set up retriever tool."""
    
    current_checksum = generate_patient_data_checksum(patient_id)
    collection_name = f"patient_{patient_id}"
    
    try:
        # Try to load existing collection
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory="./chroma_db"
        )
        
        # Check if checksum matches
        stored_checksum = vectorstore._collection.metadata.get("checksum") if vectorstore._collection.metadata else None
        
        print(f"🔍 Current checksum: {current_checksum[:12]}...")
        print(f"🔍 Stored checksum: {stored_checksum[:12] if stored_checksum else 'None'}...")
        
        if stored_checksum == current_checksum:
            print("✅ Using existing embeddings (no changes detected)")
            return vectorstore.as_retriever(**RETRIEVER_CONFIG)
        else:
            print("🔄 Checksum mismatch - rebuilding...")
            # Delete the existing collection to start fresh
            vectorstore._client.delete_collection(collection_name)
            
    except Exception as e:
        print(f"📝 Exception loading collection: {e}")
    
    # Need to rebuild - load documents and create embeddings
    print("🔄 Building new embeddings...")
    doc_splits = load_documents(patient_id)
    print(f"� Loaded {len(doc_splits)} chunks")
    
    print("🔍 Creating embeddings...")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    
    # Set the metadata after creation
    vectorstore._collection.modify(metadata={"checksum": current_checksum})
    
    print("✅ Retrieval tool ready")
    return vectorstore.as_retriever(**RETRIEVER_CONFIG)