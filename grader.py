"""
Document Grader for Medical RAG System
Grades documents using a binary score for relevance check.
"""

from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

# Load shared configuration (includes dotenv loading)
import config

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

grader_model = init_chat_model("openai:gpt-4o", temperature=0)

def grade_documents(
    state: MessagesState,
) -> int:
    """Determine whether the retrieved documents are relevant to the question.
    Returns:
        1 if documents are relevant (generate answer)
        0 if documents are not relevant (rewrite question)
    """
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return 1  # Documents are relevant
    else:
        return 0  # Documents are not relevant
