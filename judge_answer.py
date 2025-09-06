from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from custom_state import MedicalRAGState
from typing import Dict, Any



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


golden_reference_answers = {
    "fatigue_001": (
        "The patient's fatigue is multifactorial, primarily due to:\n\n"
        "1. **Hypothyroidism** — Lab results show TSH 8.2 (high) and Free T4 0.6 (low), "
        "which is diagnostic for primary hypothyroidism. This condition directly causes fatigue, "
        "sluggishness, cold intolerance, and weight gain.\n\n"
        "2. **Vitamin D Deficiency** — Level of 22 ng/mL (below the normal range of 30–100). "
        "Low vitamin D contributes to muscle weakness, low mood, and tiredness.\n\n"
        "3. **Prediabetes** — HbA1c 5.9% places the patient in the prediabetic range. "
        "This can cause fluctuations in blood glucose and energy crashes, worsening fatigue.\n\n"
        "4. **Lifestyle & Sleep Factors** — Patient reports ~6 hours of sleep/night, "
        "a sedentary job, inconsistent exercise, and a diet high in processed carbs and fats, "
        "all of which contribute to low energy.\n\n"
        "5. **Systemic Inflammation** — hs-CRP 3.6 mg/L (above 3.0 is high risk). "
        "Chronic low-grade inflammation is associated with increased fatigue and malaise.\n\n"
        "Clinical Impression: The fatigue is most likely driven by untreated hypothyroidism, "
        "but is compounded by vitamin D deficiency, prediabetes, poor sleep/diet, and elevated hs-CRP. "
        "Addressing hypothyroidism with Levothyroxine, correcting vitamin D levels, improving lifestyle, "
        "and monitoring metabolic risk should improve symptoms.\n\n"
        "-------------------------------------------\n"
        "IDEAL CONTEXT (must be present to generate a good answer):\n"
        "- Thyroid labs: TSH elevated, Free T4 low (diagnostic for hypothyroidism)\n"
        "- Vitamin D level (low)\n"
        "- HbA1c showing prediabetes\n"
        "- hs-CRP elevated (systemic inflammation)\n"
        "- Patient's sleep habits (~6 hrs/night, not restorative)\n"
        "- Patient's lifestyle: sedentary work, low exercise, diet high in processed foods\n"
        "- Medical history: hypercholesterolemia, hypothyroidism, vitamin D deficiency, prediabetes\n"
        "- Family history (CV risk, thyroid issues)\n"
        "- Reported symptoms: fatigue, weight gain, sluggishness, cold intolerance\n"
    ),
    "diabetes_001": (
        "Based on the available laboratory results, the patient does not currently have diabetes but is in the prediabetic range:\n\n"
        "**Current Status: Prediabetes**\n"
        "- HbA1c: 5.9% (prediabetic range: 5.7-6.4%)\n"
        "- This indicates average blood glucose levels over the past 2-3 months are elevated but not yet diabetic\n\n"
        "**Risk Factors Present:**\n"
        "- Metabolic syndrome components: elevated hs-CRP, hypercholesterolemia\n"
        "- Lifestyle factors: sedentary work, inconsistent exercise, processed food diet\n"
        "- Age and weight status\n\n"
        "**Recommendations:**\n"
        "- Lifestyle interventions (diet modification, regular exercise)\n"
        "- Regular monitoring with repeat HbA1c in 3-6 months\n"
        "- Address other metabolic risk factors\n\n"
        "IDEAL CONTEXT (must be present to generate a good answer):\n"
        "- HbA1c value (5.9%)\n"
        "- Understanding of prediabetes range (5.7-6.4%)\n"
        "- Current medications (no diabetes medications listed)\n"
        "- Risk factors: lifestyle, other metabolic markers\n"
    ),
}


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