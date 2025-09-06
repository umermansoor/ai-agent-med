
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
import glob

import config

def create_retriever(patient_id) -> VectorStoreRetriever:
    """Load medical documents, create embeddings, store in a vector store, and set up retriever tool."""
    
    # Load and process medical documents for the patient
    documents = []
    markdown_files = glob.glob(f"data/{patient_id}/**/*.md", recursive=True)
    
    for file_path in markdown_files:
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    
    print(f"📚 Loaded {len(documents)} medical documents")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(documents)
    print(f"📝 Split into {len(doc_splits)} chunks")
    
    # Create vector store and retriever
    print("🔍 Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=embeddings
    )
    
    # Configure retriever to return top 3 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    print(f"📊 Retriever configured to return top 3 most relevant chunks")
    
    
    print("✅ Retrieval tool ready")
    return retriever