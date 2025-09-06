"""
Document Grader for Medical RAG System
Evaluates whether retrieved documents are relevant to the medical question.
"""

from langchain.chat_models import init_chat_model
from custom_state import MedicalRAGState
from typing import Dict, Any, Literal
import config

GRADER_PROMPT = """
You are a medical document relevance grader. Your task is to assess whether retrieved medical documents 
contain information relevant to answering the given medical question.

Question: {question}

Retrieved Documents:
{documents}

Instructions:
1. Analyze if the retrieved documents contain medical information that could help answer the question
2. Consider medical context, patient data, lab results, symptoms, medications, etc.
3. Return ONLY "yes" if documents are relevant, "no" if not relevant

Decision (yes/no):"""

# Initialize the grader model
grader_model = init_chat_model("openai:gpt-4o", temperature=0)

def grade_documents(state: MedicalRAGState) -> Literal[1, 0]:
    """
    Determines whether the retrieved documents are relevant to the question.
    
    Args:
        state: Current state containing messages
        
    Returns:
        1 if relevant documents (proceed to generate_answer)
        0 if not relevant (proceed to rewrite_question)
    """
    question = state["messages"][0].content
    # Get documents from the last message (retrieval results)
    documents = state["messages"][-1].content if len(state["messages"]) > 1 else ""
    
    prompt = GRADER_PROMPT.format(question=question, documents=documents)
    response = grader_model.invoke([{"role": "user", "content": prompt}])
    
    # Parse the response
    grade = response.content.strip().lower()
    
    if "yes" in grade:
        return 1  # Relevant - proceed to answer generation
    else:
        return 0  # Not relevant - rewrite question and retry
