from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from custom_state import MedicalRAGState
from typing import Dict, Any
import json
import os



JUDGE_PROMPT = """
You are a judge assessing the quality of a medical answer against golden reference answers.

Original Question:
{question}

System Generated Answer:
{generated_answer}

System Context:
{context}

Golden Reference Answer:
{golden_answer}

Your job:
A) Compare SYSTEM_ANSWER to GOLDEN_ANSWER for medical accuracy and relevance.
B) Evaluate SYSTEM_CONTEXT against the IDEAL CONTEXT requirements and penalize each missing item.


--------------------------------
INSTRUCTIONS (READ CAREFULLY)
--------------------------------

1) Parse IDEAL CONTEXT
   - If GOLDEN_ANSWER contains a section titled 'IDEAL CONTEXT' (bulleted list), extract each bullet as a required context item.
   - If no explicit IDEAL CONTEXT is present, infer key required items from GOLDEN_ANSWER (labs, vitals, diagnoses, symptoms, meds, lifestyle factors) and treat them as the checklist.

2) Identify Context Coverage
   - For each required context item, check if SYSTEM_CONTEXT contains it (exact value or a clear equivalent).
   - Mark each item as PRESENT or MISSING.
   - Count MISSING items.

3) Grade the Answer (Answer Score: 1–10)
   - Start from 10.
   - Subtract 1–2 points for each clinically important discrepancy vs GOLDEN_ANSWER (incorrect fact, harmful advice, omissions of major causes).
   - Subtract 0.5–1 point for minor omissions or lack of clarity.
   - Never go below 1.

4) Grade the Context (Context Score: 1–10)
   - Start from 10.
   - Penalize EACH missing required context item by 1 point.
     * If an item is partially present (e.g., “thyroid abnormal” but missing the specific TSH/T4 values), subtract 0.5.
   - If >8 items are missing, cap score at 1.
   - If critical safety items are missing (e.g., key diagnostic labs directly tied to the question), subtract an additional 1 (once total).

5) Feedback Requirements (most important part):
   - Explicitly list which required context items were PRESENT and which were MISSING.
   - Explain how missing context likely affected SYSTEM_ANSWER quality.
   - Identify major content differences between SYSTEM_ANSWER and GOLDEN_ANSWER, noting any inaccuracies or omissions.

6) Output Format (use exactly this):
Answer Score: <1–10>
Context Score: <1–10>
Feedback:
- Similarities/Differences vs Golden Answer: <analysis>
- Context Coverage:
  - Present: <comma-separated checklist items>
  - Missing: <comma-separated checklist items>
- Impact of Missing Context on the Answer: <analysis>

Notes:
- Prefer concrete, clinical specifics (e.g., "TSH 8.2, Free T4 0.6") over vague mentions.
- If SYSTEM_ANSWER is good but SYSTEM_CONTEXT is incomplete, reflect that in the Context Score.
- If the IDEAL CONTEXT has duplicate/overlapping items, merge them reasonably to avoid double-penalizing.
"""


def load_golden_answers(patient_id: str = "drapoel"):
    """Load golden answers from the JSON file."""
    golden_file = f"golden_data/{patient_id}/golden.json"
    if not os.path.exists(golden_file):
        return {}
    
    try:
        with open(golden_file, 'r') as f:
            data = json.load(f)
        
        # Extract golden answers and format them for the judge
        golden_answers = {}
        for question_id, question_data in data.get("questions", {}).items():
            if "golden_answer" in question_data:
                golden_content = question_data["golden_answer"]["content"]
                ideal_context = question_data["golden_answer"].get("ideal_context", [])
                
                # Format like the old structure with IDEAL CONTEXT section
                if ideal_context:
                    golden_content += "\n\n-------------------------------------------\n"
                    golden_content += "IDEAL CONTEXT (must be present to generate a good answer):\n"
                    for context_item in ideal_context:
                        golden_content += f"- {context_item}\n"
                
                golden_answers[question_id] = golden_content
        
        return golden_answers
    except Exception as e:
        print(f"Error loading golden answers: {e}")
        return {}


# Load golden answers from JSON file instead of hardcoding them
golden_reference_answers = load_golden_answers("drapoel")


def get_question_id_from_text(question_text: str) -> str:
    """Map question text to question ID for backward compatibility."""
    question_mapping = {
        "why is this patient feeling tired?": "fatigue_001",
        "does the patient have diabetes?": "diabetes_001",
        # Add more mappings as needed
    }
    return question_mapping.get(question_text.lower(), "unknown")


def judge_answer(state: MedicalRAGState) -> Dict[str, Any]:
    """Judge the quality of the generated answer against the golden reference answer using ID-based matching."""
    # Get question ID from state, with fallback to text-based matching
    question_id = state.get("question_id")
    if not question_id:
        # Fallback for backward compatibility
        question_text = state["messages"][0].content
        question_id = get_question_id_from_text(question_text)
    
    original_question = state.get("original_question", state["messages"][0].content)
    generated_answer = state["messages"][-1].content
    context = state.get("retrieved_context", state["messages"][-2].content if len(state["messages"]) > 1 else "")

    # Use ID-based lookup for golden answer
    golden_answer = golden_reference_answers.get(question_id, "No golden reference available for this question ID.")

    prompt = JUDGE_PROMPT.format(
        question=original_question,
        generated_answer=generated_answer,
        context=context,
        golden_answer=golden_answer
    )

    judge_model = init_chat_model("openai:gpt-4o", temperature=0)
    response = judge_model.invoke([{"role": "user", "content": prompt}])

    return {"messages": [response]}