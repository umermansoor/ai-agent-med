from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from custom_state import MedicalRAGState
from typing import Dict, Any
import json
import os


CONTEXT_JUDGE_PROMPT = """
You are a judge evaluating whether the RAG retrieval context contains the necessary information within the context to answer the question correctly. Your goal is to focus on the quality of the retrieved context and whether or not it contains the information needed to answer the question accurately, based on a golden context.

Original Question:
{question}

System Retrieved Context:
{context}

Golden Reference Answer:
{golden_answer}

Ideal Context Requirements:
{ideal_context}

Focus only on whether the information needed to answer correctly is available in the context, not on how well the LLM used it.
For example, if the golden answer requires knowing a medication name and dosage, the context must include both exactly as in the golden answer.
If the newer information overrides older info, the context must include the newer info.
You will also penalize if the context contains too much irrelevant information that could confuse the LLM.

Evaluation Rules:
1) If the GOLDEN_ANSWER or 'IDEAL CONTEXT' are missing, do not judge the answer and return "No golden reference available for this question ID." Scores are -1
2) Start from 10 and subtract points for missing or incomplete required items:
    - Context is missing information that's present in the ideal context. -2 points for each.
    - Context contains more than 50% irrelevant information that doesn't help answer the question. -2 points total.
    - If a more recent document clearly updates a fact and SYSTEM_CONTEXT omits it, subtract an additional -2 once total.
3) Floor: minimum Context Score = 0.
"""

ANSWER_JUDGE_PROMPT = """
You are a judge assessing whether a medical answer reports **accurate, patient-specific facts** compared with a golden reference. If the answer contains all the required information, even if the language is not perfect, give full points.

Original Question:
{question}

System Generated Answer:
{generated_answer}

Golden Reference Answer:
{golden_answer}

Evaluation Rules:
- Start from 10 and penalize heavily for factual errors or omissions:
    - Missing a REQUIRED information present in the golden answer: **-2 each**
    - Do NOT penalize for minor language issues if all required info is present and would make sens to a medical professional.
    - Floor: minimum Answer Score = 1
"""

def load_golden_answers(patient_id: str = "drapoel"):
    """Load golden answers from the JSONL file."""
    golden_file = f"golden_data/{patient_id}/golden.jsonl"
    if not os.path.exists(golden_file):
        return {}
    
    try:
        golden_answers = {}
        with open(golden_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                question_id = data.get("id")
                golden_answer = data.get("golden_answer", {})
                
                if not (question_id and golden_answer):
                    continue
                    
                content = golden_answer.get("content", [])
                formatted_content = "\n".join([f"- {item}" for item in content]) if isinstance(content, list) else str(content)
                
                golden_answers[question_id] = {
                    "content": formatted_content,
                    "ideal_context": golden_answer.get("ideal_context", [])
                }
        
        return golden_answers
    except Exception as e:
        print(f"Error loading golden answers: {e}")
        return {}


# Load golden answers from JSONL file
golden_reference_answers = load_golden_answers("drapoel")


def _get_judgment_data(state: MedicalRAGState):
    """Helper function to extract common judgment data from state."""
    question_id = state.get("question_id")
    if not question_id:
        return None, "No question ID available for judgment."
    
    golden_data = golden_reference_answers.get(question_id)
    if not golden_data:
        return None, "No golden reference available for this question ID."
    
    original_question = state.get("original_question", state["messages"][0].content)
    return {
        "question_id": question_id,
        "original_question": original_question,
        "golden_data": golden_data
    }, None


def _invoke_judge_model(prompt: str) -> str:
    """Helper function to invoke the judgment model."""
    judge_model = init_chat_model("openai:gpt-4o", temperature=0)
    response = judge_model.invoke([{"role": "user", "content": prompt}])
    return response.content


def judge_context(state: MedicalRAGState) -> Dict[str, Any]:
    """Judge the quality of the retrieved context against ideal context requirements."""
    data, error = _get_judgment_data(state)
    if error:
        return {"context_judgment": error}
    
    context = state.get("retrieved_context", state["messages"][-2].content if len(state["messages"]) > 1 else "")
    
    ideal_context = "\n".join([f"- {item}" for item in data["golden_data"]["ideal_context"]])

    prompt = CONTEXT_JUDGE_PROMPT.format(
        question=data["original_question"],
        context=context,
        golden_answer=data["golden_data"]["content"],
        ideal_context=ideal_context
    )

    return {"context_judgment": _invoke_judge_model(prompt)}


def judge_answer_accuracy(state: MedicalRAGState) -> Dict[str, Any]:
    """Judge the accuracy of the generated answer against the golden reference."""
    data, error = _get_judgment_data(state)
    if error:
        return {"answer_judgment": error}
    
    generated_answer = state["messages"][-1].content

    prompt = ANSWER_JUDGE_PROMPT.format(
        question=data["original_question"],
        generated_answer=generated_answer,
        golden_answer=data["golden_data"]["content"]
    )

    return {"answer_judgment": _invoke_judge_model(prompt)}


def judge_answer(state: MedicalRAGState) -> Dict[str, Any]:
    """Combined judge that runs both context and answer evaluations."""
    context_result = judge_context(state)
    answer_result = judge_answer_accuracy(state)
    
    combined_feedback = f"""=== CONTEXT EVALUATION ===
{context_result.get('context_judgment', 'No context judgment available')}

=== ANSWER EVALUATION ===
{answer_result.get('answer_judgment', 'No answer judgment available')}"""
    
    return {"messages": [AIMessage(content=combined_feedback)]}
