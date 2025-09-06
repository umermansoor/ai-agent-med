"""
Custom State Schema for Medical RAG System
This extends MessagesState to include question_id for better judge matching
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class MedicalRAGState(TypedDict):
    """Custom state schema that includes question ID for precise judge matching."""
    
    # Standard messages field with reducer
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Custom field for question identification
    question_id: str
    
    # Optional: Store original question text for reference
    original_question: str
    
    # Optional: Store context for judge evaluation
    retrieved_context: str


# Alternative with optional fields
class MedicalRAGStateOptional(TypedDict, total=False):
    """Custom state schema with optional fields."""
    
    # Required field
    messages: Annotated[list[BaseMessage], add_messages]
    question_id: str
    
    # Optional fields
    original_question: str
    retrieved_context: str
    patient_id: str
    confidence_score: float
