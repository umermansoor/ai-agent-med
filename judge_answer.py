from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from custom_state import MedicalRAGState
from golden_data_loader import load_golden_answers_formatted
from typing import Dict, Any
import json
import os



JUDGE_PROMPT = """
You are a judge assessing whether a medical answer reports **accurate, patient-specific facts** compared with a golden reference.

Priority: Accuracy of concrete patient data (medications, dosages, frequencies, lab values, diagnoses, and other explicitly documented information). Do **not** reward speculation or predictions.

Original Question:
{question}

System Generated Answer:
{generated_answer}

System Context:
{context}

Golden Reference Answer:
{golden_answer}

--------------------------------
EVALUATION RULES
--------------------------------

1) Parse IDEAL CONTEXT
   - If the GOLDEN_ANSWER or 'IDEAL CONTEXT' are missing, do not judge the answer and return "No golden reference available for this question ID." Scores are -1 for both Answer Score and Context Score.
   - If GOLDEN_ANSWER includes an 'ideal_context' checklist, treat each bullet as REQUIRED.
   - If not present, infer REQUIRED facts from GOLDEN_ANSWER (e.g., exact medication names + doses + frequencies, lab values, explicit diagnoses).
   - REQUIRED items may also include **document sources** if they are listed in IDEAL CONTEXT.

2) Fact Checking (Answer Score)
   Start from 10 and penalize heavily for factual errors or omissions:
   - Missing a REQUIRED medication: **-3 each**
   - Wrong medication listed (not in golden answer): **-3 each**
   - Wrong dosage or frequency: **-2 each**
   - Wrong lab value or threshold (when explicit): **-2 each**
   - Stating or implying a condition not supported by explicit data, or “predicting” something already clearly documented: **-3**
   - Adding minor extras not requested (e.g., listing supplements when the question says not to): **-0.5 each** (cap -2 total)
   - Using vague language when specifics are available (e.g., “thyroid medicine” instead of “Levothyroxine 100 mcg daily”): **-1**
   Floor: minimum Answer Score = 1.

3) Context Adequacy (Context Score)
   Start from 10 and check SYSTEM_CONTEXT against REQUIRED items:
   - Missing REQUIRED item entirely: **-1 each**
   - Present but incomplete (e.g., mentions medication without dose/frequency, or dose outdated vs most recent doc): **-0.5**
   - If a more recent document clearly updates a fact and SYSTEM_CONTEXT omits it, subtract an additional **-1 once total**.
   Floor: minimum Context Score = 1.

4) Output Format (must use exactly):
Answer Score: <1-10>
Context Score: <1-10>
Feedback:
- Critical Errors/Misses: <bullet list of the highest-impact inaccuracies or omissions>
- Extras Not Requested: <list, if any>
- Context Coverage:
  - Present: <comma-separated REQUIRED items found>
  - Missing/Incomplete: <comma-separated REQUIRED items missing or partial>
- Rationale: <brief explanation tying penalties to rules above>

--------------------------------
NOTES
--------------------------------
- Prefer exact specifics (e.g., “Levothyroxine 100 mcg daily”) over generalities.
- Heavily penalize missing or incorrect **patient data** over style/wording issues.
- Do not give credit for reasoning or guesses beyond what is explicitly documented.
- Always consider **document recency** when determining which version of a fact is correct.
"""



# Load golden answers using the shared utility
golden_reference_answers = load_golden_answers_formatted("drapoel")


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