"""
Reranked Retriever - A wrapper that adds Cohere reranking to any LangChain retriever.
"""

from reranker import CohereReranker


class RerankedRetriever:
    """Simple wrapper that adds reranking to any retriever."""
    
    def __init__(self, base_retriever, top_k=5, verbose=True):
        self.retriever = base_retriever
        self.reranker = CohereReranker()
        self.top_k = top_k
        self.verbose = verbose
    
    def get_relevant_documents(self, query: str):
        """Get documents and rerank them."""
        # Get initial documents
        initial_docs = self.retriever.invoke(query)
        
        if self.verbose:
            print(f"🔍 Retrieved {len(initial_docs)} initial documents")
        
        # Remove duplicates based on content
        unique_docs = self._deduplicate_documents(initial_docs)
        
        if self.verbose:
            print(f"📝 Deduplicated to {len(unique_docs)} unique documents")
        
        if not unique_docs:
            return []
        
        # Rerank documents
        reranked_docs = self._rerank_documents(query, unique_docs)
        
        if self.verbose:
            print(f"📊 Reranked to top {len(reranked_docs)} documents")
        
        return reranked_docs
    
    def _deduplicate_documents(self, docs):
        """Remove duplicate documents based on content hash."""
        unique_docs = []
        seen_hashes = set()
        
        for doc in docs:
            content_hash = hash(doc.page_content[:200].strip())
            if content_hash not in seen_hashes:
                unique_docs.append(doc)
                seen_hashes.add(content_hash)
        
        return unique_docs
    
    def _rerank_documents(self, query, docs):
        """Rerank documents using Cohere."""
        # Extract text for reranking
        documents_text = [doc.page_content for doc in docs]
        
        # Get reranked results
        reranked_results = self.reranker.rerank(query, documents_text, top_k=self.top_k)
        
        # Map back to original documents
        reranked_docs = []
        for text, score in reranked_results:
            # Find matching document
            for doc in docs:
                if doc.page_content == text:
                    # Add rerank score to metadata
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['rerank_score'] = score
                    reranked_docs.append(doc)
                    break
        
        return reranked_docs
    
    def invoke(self, inputs, config=None):
        """Compatibility method for LangChain integration."""
        if isinstance(inputs, dict) and 'query' in inputs:
            return self.get_relevant_documents(inputs['query'])
        elif isinstance(inputs, str):
            return self.get_relevant_documents(inputs)
        else:
            return self.get_relevant_documents(str(inputs))
