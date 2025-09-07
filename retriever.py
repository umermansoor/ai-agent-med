
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from reranked_retriever import RerankedRetriever
import glob
import hashlib

import config

# Configuration
RETRIEVER_CONFIG = {
    "search_type": "similarity",  # Use similarity search for better scoring
    "search_kwargs": {"k": 25}    # Get 25 documents from 104 total chunks for reranking
}

def generate_patient_data_checksum(patient_id: str) -> str:
    """Generate checksum for all patient data files."""
    hasher = hashlib.sha256()
    markdown_files = sorted(glob.glob(f"data/{patient_id}/**/*.md", recursive=True))
    
    for file_path in markdown_files:
        hasher.update(file_path.encode())
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
    
    return hasher.hexdigest()


def load_documents(patient_id: str):
    """Load and split patient documents."""
    # Load all markdown files
    documents = []
    for file_path in glob.glob(f"data/{patient_id}/**/*.md", recursive=True):
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=250, 
        separators=["\n---\n", "\n# ", "\n## ", "\n"]
    )
    return text_splitter.split_documents(documents)

def create_retriever(patient_id) -> RerankedRetriever:
    """Create a retriever with reranking for the given patient."""
    current_checksum = generate_patient_data_checksum(patient_id)
    collection_name = f"patient_{patient_id}"
    
    # Try to load existing collection
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory="./chroma_db"
        )
        
        stored_checksum = vectorstore._collection.metadata.get("checksum") if vectorstore._collection.metadata else None
        
        print(f"🔍 Checksum - Current: {current_checksum[:12]}... | Stored: {stored_checksum[:12] if stored_checksum else 'None'}...")
        
        if stored_checksum == current_checksum:
            print("✅ Using existing embeddings")
        else:
            print("🔄 Rebuilding embeddings...")
            vectorstore._client.delete_collection(collection_name)
            raise Exception("Checksum mismatch - rebuild needed")
            
    except Exception:
        # Build new embeddings
        print("🔄 Loading documents...")
        doc_splits = load_documents(patient_id)
        print(f"📄 Processing {len(doc_splits)} chunks...")
        
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        vectorstore._collection.modify(metadata={"checksum": current_checksum})
        print("✅ Embeddings ready")
    
    # Create base retriever and wrap with reranking
    base_retriever = vectorstore.as_retriever(**RETRIEVER_CONFIG)
    return RerankedRetriever(base_retriever, top_k=5)