"""
Context Compressor for Medical RAG System
Uses an LLM to remove clearly irrelevant information from retrieved context while being very conservative.
"""

from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from custom_state import MedicalRAGState
from typing import Dict, Any

# Load shared configuration (includes dotenv loading)
import config

COMPRESS_PROMPT = """
You are a medical context compressor. Your job is to review the retrieved medical context and ONLY remove information that you are HIGHLY CONFIDENT is completely irrelevant to answering the medical question.

Be EXTREMELY CONSERVATIVE - when in doubt, KEEP the information. Only remove sections if you are absolutely certain they cannot possibly help answer the question.

Medical Question: {question}

Retrieved Context:
{context}

Instructions:
1. Carefully analyze what information is needed to answer the medical question
2. ONLY remove information that is clearly unrelated (e.g., if question is about medications, you might remove detailed family history of unrelated conditions, but keep any family history that might relate to medication needs)
3. Keep ALL information that might be even remotely relevant
4. Preserve exact medical details, dosages, dates, lab values
5. Maintain the structure and formatting of medical data
6. If a section contains both relevant and irrelevant information, keep the entire section

Examples of what to potentially remove (ONLY if highly confident):
- Detailed family history when question is about current lab values (but keep if it might explain genetic predispositions)
- Allergy information when question is about imaging results (but keep if medications are involved)
- Supplement details when question specifically excludes supplements (but be careful of interactions)

Remember: It's better to keep too much information than to accidentally remove something important. When in doubt, KEEP IT.

Return ONLY the compressed context, maintaining all formatting:
"""

# Initialize the compressor model
compressor_model = init_chat_model("openai:gpt-4o", temperature=0)

def compress_context(state: MedicalRAGState) -> Dict[str, Any]:
    """Compress retrieved context by removing clearly irrelevant information."""
    question = state["messages"][0].content
    current_context = state["messages"][-1].content
    
    print(f"\n🗜️  Compressing context...")
    print(f"📏 Original context length: {len(current_context)} characters")
    
    prompt = COMPRESS_PROMPT.format(
        question=question,
        context=current_context
    )
    
    response = compressor_model.invoke([{"role": "user", "content": prompt}])
    compressed_context = response.content
    
    compression_ratio = len(compressed_context) / len(current_context)
    print(f"📏 Compressed context length: {len(compressed_context)} characters")
    print(f"📊 Compression ratio: {compression_ratio:.2%}")
    
    # Replace the last message (retrieved context) with compressed version
    return {"messages": [AIMessage(content=compressed_context)]}
