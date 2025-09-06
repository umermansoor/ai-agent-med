from retriever import create_retriever
from grader import grade_documents
from rewriter import rewrite_question
from generate_answer import generate_answer

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

import config

from langchain.tools.retriever import create_retriever_tool


llm_model = init_chat_model("openai:gpt-4o", temperature=0)
retriever = create_retriever("drapoel")

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_patient_health_data",
    "Search patient health records and return relevant information."
)

def run_retrieval_or_respond(state: MessagesState):
    """Decide whether to retrieve documents or respond directly based on the question. 
    E.g. if the user asks "what is flu virus?", the LLM can respond directly without retrieval.
    """
    response = llm_model.bind_tools([retriever_tool]).invoke(state["messages"])
    
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node(run_retrieval_or_respond) # first node 
workflow.add_node("retrieve", ToolNode([retriever_tool])) 
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "run_retrieval_or_respond")

# Tell LangGraph how to route based on the output of run_retrieval_or_respond
workflow.add_conditional_edges(
    "run_retrieval_or_respond",
    tools_condition,
    {
        "tools": "retrieve", # if LLM decides to use tool, go to retrieve node
        END: END, # if LLM responds directly, go to END
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        1: "generate_answer",      # If grader returns 1 (relevant) → generate answer
        0: "rewrite_question",     # If grader returns 0 (not relevant) → rewrite question
    },
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "run_retrieval_or_respond")

# Compile
graph = workflow.compile()

for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Why is this patient feeling tired?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")