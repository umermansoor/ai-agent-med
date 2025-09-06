"""
Medical Question Rewriter for RAG System
Rewrites user questions to be more specific and medical-focused.
"""

from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# Load shared configuration (includes dotenv loading)
import config

from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# Load shared configuration
import config

REWRITE_PROMPT = (
    "You are a medical question rewriter. Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "The available medical data includes: patient intake information, medications (prescriptions and supplements), "
    "laboratory results (CBC, CMP, lipid panel, thyroid function, A1C, vitamin D, urinalysis, hs-CRP), "
    "imaging studies (body scans), and genetic information.\n\n"
    "Consider these medical data categories when rewriting:\n"
    "- Demographics: age, gender, ethnicity, medical history\n"
    "- Medications: current prescriptions, supplements, dosages, indications\n"
    "- Laboratory values: blood work, metabolic panels, cardiac markers, vitamin levels\n"
    "- Clinical findings: symptoms, physical exam findings, vital signs\n"
    "- Imaging: radiological studies, scan results\n"
    "- Genetics: genetic variants, hereditary conditions\n\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Rewrite this question to be more specific and use proper medical terminology that would help retrieve relevant medical information. "
    "Focus on the specific data type most relevant to the question (labs, medications, imaging, etc.).\n"
    "Formulate an improved medical question:"
)

# Initialize the rewriter model
rewriter_model = init_chat_model("openai:gpt-4o", temperature=0)

def rewrite_question(state: MessagesState):
    """Rewrite the original user question to be more medically specific."""
    messages = state["messages"]
    question = messages[0].content
    
    prompt = REWRITE_PROMPT.format(question=question)
    response = rewriter_model.invoke([{"role": "user", "content": prompt}])
    
    # Return a proper HumanMessage object instead of dictionary
    return {"messages": [HumanMessage(content=response.content)]}
