from retriever import create_retriever
from grader import grade_documents
from rewriter import rewrite_question
from generate_answer import generate_answer
from judge_answer_split import judge_answer
from custom_state import MedicalRAGState
from golden_data_loader import load_golden_questions_raw

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.chat_models import init_chat_model

import config
import json
import os
import argparse
from datetime import datetime

from langchain.tools.retriever import create_retriever_tool


def create_workflow(use_reranker=True):
    """Create a fresh workflow instance.
    
    Args:
        use_reranker: If True, use reranked retriever. If False, use base retriever only.
    """
    llm_model = init_chat_model("openai:gpt-4o", temperature=0)
    retriever = create_retriever("drapoel", use_reranker=use_reranker)

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_patient_health_data",
        "Search the current patient's health records using Semantic Search (RAG) and return relevant information. Always use this tool when asked about patient data - no patient ID needed."
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


def run_single_question(question_data: dict, use_reranker=True):
    """Run a single question through the workflow with fresh state.
    
    Args:
        question_data: Dictionary containing question data
        use_reranker: If True, use reranked retriever. If False, use base retriever only.
    """
    print(f"\n{'='*80}")
    print(f"🔍 Question ID: {question_data['id']}")
    print(f"📝 Question: {question_data['text']}")
    print(f"🔧 Reranker: {'Enabled' if use_reranker else 'Disabled'}")
    print(f"{'='*80}")
    
    # Create fresh workflow instance
    graph = create_workflow(use_reranker=use_reranker)
    
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
    
    # Variables to capture results
    system_answer = None
    judge_feedback = None
    
    step_count = 0
    # Run the workflow
    for chunk in graph.stream(input_state):
        for node, update in chunk.items():
            step_count += 1
            print(f"\n🔄 Step {step_count}: Update from node '{node}'")
            if "messages" in update and update["messages"]:
                try:
                    update["messages"][-1].pretty_print()
                    
                    # Capture system answer from generate_answer node
                    if node == "generate_answer":
                        system_answer = update["messages"][-1].content
                    
                    # Capture complete judge feedback from judge_answer node
                    elif node == "judge_answer":
                        judge_feedback = update["messages"][-1].content
                        
                except Exception as e:
                    print(f"Content: {update['messages'][-1].content}")
            print("-" * 40)
    
    # Save results to file
    save_results(question_data, system_answer, judge_feedback)
    
    print(f"✅ Completed question: {question_data['id']}")
    return True


def save_results(question_data: dict, system_answer: str, judge_feedback: str):
    """Save question, system answer, and complete judge feedback to results.txt"""
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"QUESTION ID: {question_data['id']}\n")
        f.write(f"QUESTION: {question_data['text']}\n")
        f.write(f"\nSYSTEM ANSWER:\n{system_answer or 'No answer captured'}\n")
        f.write(f"\nJUDGE EVALUATION:\n{judge_feedback or 'No feedback captured'}\n")
        f.write(f"{'='*80}\n\n")


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Medical RAG Agent')
    parser.add_argument('--no-reranker', action='store_true', 
                       help='Disable reranker and use base retriever only')
    args = parser.parse_args()
    
    use_reranker = not args.no_reranker
    
    try:
        # Clear results file at start
        with open("results.txt", "w", encoding="utf-8") as f:
            f.write(f"MEDICAL RAG EVALUATION RESULTS\n")
            f.write(f"Generated on: {json.dumps(str(datetime.now()))}\n")
            f.write(f"Reranker: {'Enabled' if use_reranker else 'Disabled'}\n")
            f.write(f"{'='*80}\n\n")
        
        # Load golden questions
        golden_questions = load_golden_questions_raw("drapoel")
        print(f"📚 Loaded {len(golden_questions)} golden questions")
        print(f"🔧 Reranker: {'Enabled' if use_reranker else 'Disabled'}")
        
        # Run each question with fresh state
        for question_id, question_data in golden_questions.items():
            try:
                run_single_question(question_data, use_reranker=use_reranker)
            except Exception as e:
                print(f"❌ Error processing question {question_id}: {str(e)}")
                continue
        
        print(f"\n🎉 Completed processing all {len(golden_questions)} questions!")
        print(f"📄 Results saved to results.txt")
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()