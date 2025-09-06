"""
Medical Answer Generator for RAG System
Generates comprehensive medical answers from retrieved patient data.
"""

from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

# Load shared configuration (includes dotenv loading)
import config

GENERATE_PROMPT = (
    "You are a medical assistant helping healthcare professionals analyze patient data. "
    "Use the following pieces of retrieved patient information to answer the medical question. "
    "Provide accurate, clinical information based only on the available data. "
    "If the information is not available in the provided context, clearly state that. "
    "Use professional medical terminology but keep explanations clear and concise. "
    "Limit your response to three sentences maximum.\n\n"
    "Medical Question: {question}\n"
    "Patient Data Context: {context}\n\n"
    "Medical Answer:"
)

# Initialize the health expert model
expert_model = init_chat_model("openai:gpt-4o", temperature=0)

def generate_answer(state: MessagesState):
    """Generate a medical answer based on patient data."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = expert_model.invoke([{"role": "user", "content": prompt}])
    
    return {"messages": [response]}
