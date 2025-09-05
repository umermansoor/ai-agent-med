#!/usr/bin/env python3
"""
Medical RAG - Tutorial Step 2: Create a retriever tool
"""

import glob
from pathlib import Path
import os
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

# Load shared configuration (includes dotenv loading)
import config

# Import the grader, rewriter, and answer generator
from grader import grade_documents, GradeDocuments
from rewriter import rewrite_question
from generate_answer import generate_answer

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

def main():
    """Tutorial Step 2: Create a retriever tool"""
    
    # Load documents
    documents = []
    markdown_files = glob.glob("data/**/*.md", recursive=True)
    
    for file_path in markdown_files:
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    
    print(f"Loaded {len(documents)} documents")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(doc_splits)} chunks")
    
    # 1. Use OpenAI embeddings (reliable hosted option)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("OPENAI_API_KEY not set. Please set it in your .env file.")
        return
    
    print("Using OpenAI embeddings")
    embeddings = OpenAIEmbeddings()
    
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=embeddings
    )
    retriever = vectorstore.as_retriever()
    
    # 2. Create a retriever tool using LangChain's prebuilt create_retriever_tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_medical_info",
        "Search and return information about patient medical data.",
    )
    
    # 3. Test the tool
    #result = retriever_tool.invoke({"query": "what allergies does the patient has?"})
    #print("\nRetriever tool result:")
    #print(result)

    response_model = init_chat_model("openai:gpt-4o", temperature=1)

    def generate_query_or_respond(state: MessagesState):
        """Call the model to generate a query or respond to the user based on the state."""
        response = (
            response_model
             .bind_tools([retriever_tool]).invoke(state["messages"])
        )
        return response
    
    # Test the tool with a sample query
    input = {"messages": [{"role": "user", "content": "what allergies does the patient has?"}]}
    response = generate_query_or_respond(input)
    response.pretty_print()

    # Test the grader with proper state structure
    print("\n" + "="*50)
    print("TESTING DOCUMENT GRADER")
    print("="*50)
    
    # Get some context from the retriever tool for testing
    docs = retriever.invoke("what allergies does the patient has?")
    context = docs[0].page_content if docs else "No documents found"
    
    # Create proper state structure for the grader
    from langchain_core.messages import HumanMessage, AIMessage
    grader_test_state = {
        "messages": [
            HumanMessage(content="what allergies does the patient has?"),
            AIMessage(content=context)
        ]
    }
    
    # Test the grader
    decision = grade_documents(grader_test_state)
    print(f"Grader decision: {decision}")
    
    # Test the rewriter
    print("\n" + "="*50)
    print("TESTING QUESTION REWRITER")
    print("="*50)
    
    # Test with various vague medical questions
    test_questions = [
        "What's wrong with the patient?",
        "Any medicines?", 
        "Blood work results?",
        "What allergies?",
        "Health problems?"
    ]
    
    for question in test_questions:
        print(f"\n📝 Original question: '{question}'")
        
        # Create state for rewriter with proper message objects
        from langchain_core.messages import HumanMessage
        rewriter_state = {
            "messages": [HumanMessage(content=question)]
        }
        
        # Rewrite the question
        rewritten_state = rewrite_question(rewriter_state)
        rewritten_question = rewritten_state["messages"][0]["content"]
        
        print(f"🔄 Rewritten question: '{rewritten_question}'")
        print("-" * 50)
    
    # Test the health expert answer generator
    print("\n" + "="*50)
    print("TESTING HEALTH EXPERT")
    print("="*50)
    
    # Test with a medical question and relevant context
    medical_question = "What prescription medications is the patient currently taking and why?"
    
    # rewrite the question first
    from langchain_core.messages import HumanMessage
    rewriter_state = {
        "messages": [HumanMessage(content=medical_question)]
    }
    rewritten_state = rewrite_question(rewriter_state)
    medical_question = rewritten_state["messages"][0]["content"]

    # grade the documents first
    from langchain_core.messages import HumanMessage, AIMessage
    grader_state = {
        "messages": [
            HumanMessage(content=medical_question),
            AIMessage(content=context)
        ]
    }
    decision = grade_documents(grader_state)
    print(f"Grader decision: {decision}")

    if decision is not "generate_answer":
        print("Documents not relevant, rewriting question again.")
    else:
        # Get relevant context from retriever
        medical_docs = retriever.invoke(medical_question)
        medical_context = medical_docs[0].page_content if medical_docs else "No medical information found"
        
        print(f"🔍 Medical Question: '{medical_question}'")
        print(f"📄 Retrieved Context: {medical_context[:200]}...")
    
        # Create state for health expert
        from langchain_core.messages import HumanMessage, AIMessage
        expert_state = {
            "messages": [
                HumanMessage(content=medical_question),
                AIMessage(content=medical_context)
            ]
        }
        
        # Generate medical answer
        answer_state = generate_answer(expert_state)
        medical_answer = answer_state["messages"][0].content
        
        print(f"🏥 Health Expert Answer: {medical_answer}")
        print("-" * 50)


if __name__ == "__main__":
    main()
