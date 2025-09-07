
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from reranker import CohereReranker
import glob
import hashlib

import config

# Retriever configuration
RETRIEVER_CONFIG = {
    "search_type": "mmr",  # Use similarity search to get scores
    "search_kwargs": {"k": 25}  # Get more documents for reranking
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
            base_retriever = vectorstore.as_retriever(**RETRIEVER_CONFIG)
            return RerankedRetriever(base_retriever, top_k=5)
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
    base_retriever = vectorstore.as_retriever(**RETRIEVER_CONFIG)
    return RerankedRetriever(base_retriever, top_k=5)
    return vectorstore.as_retriever(**RETRIEVER_CONFIG)


class RerankedRetriever:
    """Wrapper that adds reranking to any retriever."""
    
    def __init__(self, base_retriever, top_k=5):
        self.retriever = base_retriever
        self.reranker = CohereReranker()
        self.top_k = top_k
    
    def get_relevant_documents(self, query: str):
        """Get documents and rerank them."""
        # Get initial documents with similarity search using invoke method
        initial_docs = self.retriever.invoke(query)
        print(f"🔍 Retrieved {len(initial_docs)} initial documents")
        
        # Debug: Check document structure
        if initial_docs:
            print(f"📋 Document structure: {type(initial_docs[0])}")
            if hasattr(initial_docs[0], 'metadata'):
                print(f"📋 Metadata keys: {initial_docs[0].metadata.keys()}")
        
        # Log original documents with similarity scores
        for i, doc in enumerate(initial_docs[:10], 1):  # Show first 10
            # Try to get similarity score from metadata
            similarity_score = "N/A"
            if hasattr(doc, 'metadata') and doc.metadata:
                similarity_score = doc.metadata.get('score', doc.metadata.get('similarity_score', 'N/A'))
            
            print(f"  Original #{i}: similarity_score={similarity_score}")
            
        # Remove duplicates based on content similarity
        unique_docs = []
        seen_hashes = set()
        
        for doc in initial_docs:
            # Create a hash of the first 200 characters to identify similar content
            content_hash = hash(doc.page_content[:200].strip())
            if content_hash not in seen_hashes:
                unique_docs.append(doc)
                seen_hashes.add(content_hash)
        
        print(f"📝 Deduplicated to {len(unique_docs)} unique documents")
        
        if len(unique_docs) == 0:
            return []
        
        # Extract text content for reranking
        documents_text = [doc.page_content for doc in unique_docs]
        
        # Rerank documents
        reranked_results = self.reranker.rerank(query, documents_text, top_k=self.top_k)
        
        # Create mapping from original position to metadata 
        doc_mapping = {i: doc for i, doc in enumerate(unique_docs)}
        
        # Log reranking results
        print(f"📊 Reranking results (top {len(reranked_results)}):")
        
        # Create reranked document list with updated metadata
        reranked_docs = []
        for rank, (text, score) in enumerate(reranked_results, 1):
            # Find the original document that matches this text
            original_idx = None
            for i, doc in enumerate(unique_docs):
                if doc.page_content == text:
                    original_idx = i
                    break
            
            if original_idx is not None:
                original_doc = doc_mapping[original_idx]
                # Get similarity score from the original document
                similarity_score = "N/A"
                if hasattr(original_doc, 'metadata') and original_doc.metadata:
                    similarity_score = original_doc.metadata.get('score', original_doc.metadata.get('similarity_score', 'N/A'))
                
                print(f"  Rerank #{rank}: original_pos={original_idx+1}, similarity_score={similarity_score}, rerank_score={score:.4f}")
                
                # Create new document with rerank score in metadata
                reranked_doc = original_doc
                if not hasattr(reranked_doc, 'metadata'):
                    reranked_doc.metadata = {}
                reranked_doc.metadata['rerank_score'] = score
                reranked_doc.metadata['original_position'] = original_idx + 1
                reranked_docs.append(reranked_doc)
        
        return reranked_docs
    
    def invoke(self, inputs, config=None):
        """Compatibility method for LangChain integration."""
        if isinstance(inputs, dict) and 'query' in inputs:
            return self.get_relevant_documents(inputs['query'])
        elif isinstance(inputs, str):
            return self.get_relevant_documents(inputs)
        else:
            return self.get_relevant_documents(str(inputs))