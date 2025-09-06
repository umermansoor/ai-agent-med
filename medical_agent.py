from retriever import create_retriever
from grader import grade_documents
from rewriter import rewrite_question
from generate_answer import generate_answer
from judge_answer import judge_answer
from custom_state import MedicalRAGState

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.chat_models import init_chat_model

import config
import json
import os

from langchain.tools.retriever import create_retriever_tool


def load_golden_questions(patient_id: str = "drapoel"):
    """Load golden questions from the JSON file."""
    golden_file = f"golden_data/{patient_id}/golden.json"
    if not os.path.exists(golden_file):
        raise FileNotFoundError(f"Golden questions file not found: {golden_file}")
    
    with open(golden_file, 'r') as f:
        data = json.load(f)
    
    return data["questions"]


def create_workflow():
    """Create a fresh workflow instance."""
    llm_model = init_chat_model("openai:gpt-4o", temperature=0)
    retriever = create_retriever("drapoel")

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_patient_health_data",
        "Search the current patient's health records and return relevant information. Always use this tool when asked about patient data - no patient ID needed."
    )

    def run_retrieval_or_respond(state: MedicalRAGState):
        """Decide whether to retrieve documents or respond directly based on the question."""
        response = llm_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(MedicalRAGState)

    workflow.add_node(run_retrieval_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool])) 
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)
    workflow.add_node(judge_answer)

    workflow.add_edge(START, "run_retrieval_or_respond")

    workflow.add_conditional_edges(
        "run_retrieval_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            1: "generate_answer",
            0: "rewrite_question",
        },
    )

    workflow.add_edge("generate_answer", "judge_answer")
    workflow.add_edge("judge_answer", END)
    workflow.add_edge("rewrite_question", "run_retrieval_or_respond")

    return workflow.compile()


def run_single_question(question_data: dict):
    """Run a single question through the workflow with fresh state."""
    print(f"\n{'='*80}")
    print(f"🔍 Question ID: {question_data['id']}")
    print(f"📝 Question: {question_data['text']}")
    print(f"🏷️  Category: {question_data['category']} | Difficulty: {question_data['difficulty']}")
    print(f"{'='*80}")
    
    # Create fresh workflow instance
    graph = create_workflow()
    
    # Create input state
    input_state = {
        "messages": [
            {
                "role": "user",
                "content": question_data["text"],
            }
        ],
        "question_id": question_data["id"],
        "original_question": question_data["text"],
    }
    
    step_count = 0
    # Run the workflow
    for chunk in graph.stream(input_state):
        for node, update in chunk.items():
            step_count += 1
            print(f"\n🔄 Step {step_count}: Update from node '{node}'")
            if "messages" in update and update["messages"]:
                try:
                    update["messages"][-1].pretty_print()
                except Exception as e:
                    print(f"Content: {update['messages'][-1].content}")
            print("-" * 40)
    
    print(f"✅ Completed question: {question_data['id']}")
    return True


def main():
    """Main execution function."""
    try:
        # Load golden questions
        golden_questions = load_golden_questions("drapoel")
        print(f"📚 Loaded {len(golden_questions)} golden questions")
        
        # Run each question with fresh state
        for question_id, question_data in golden_questions.items():
            try:
                run_single_question(question_data)
            except Exception as e:
                print(f"❌ Error processing question {question_id}: {str(e)}")
                continue
        
        print(f"\n🎉 Completed processing all {len(golden_questions)} questions!")
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()