import cohere
import os
from typing import List, Tuple

class CohereReranker:
    def __init__(self):
        self.co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rerank documents using Cohere's rerank API."""
        if not documents:
            return []
        
        try:
            response = self.co.rerank(
                model="rerank-english-v3.0",  # Standard Cohere rerank model
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=True
            )
            
            return [(result.document.text, result.relevance_score) 
                    for result in response.results]
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}")
            # Fallback: return first top_k documents
            return [(doc, 1.0) for doc in documents[:top_k]]
