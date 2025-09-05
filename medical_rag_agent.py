#!/usr/bin/env python3
"""
Medical RAG System - Core Components
A clean implementation of retriever tool, grader, rewriter, and answer generator.
"""

import glob
from pathlib import Path
import os
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

# Load shared configuration
import config

# Import components
from grader import grade_documents
from rewriter import rewrite_question
from generate_answer import generate_answer

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

def create_medical_rag_system():
    """Create the complete medical RAG system."""
    print("🚀 Setting up Medical RAG System...")
    
    # Load and process documents
    documents = []
    markdown_files = glob.glob("data/**/*.md", recursive=True)
    
    for file_path in markdown_files:
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    
    print(f"📚 Loaded {len(documents)} medical documents")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(documents)
    print(f"📝 Split into {len(doc_splits)} chunks")
    
    # Create vector store and retriever
    print("🔍 Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=embeddings
    )
    
    # Configure retriever to return top 3 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    print(f"📊 Retriever configured to return top 3 most relevant chunks")
    
    # Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_medical_info",
        "Search and return information about patient medical data."
    )
    
    print("✅ Medical RAG System ready!")
    return retriever, retriever_tool

def test_medical_workflow():
    """Test the complete medical RAG workflow."""
    retriever, retriever_tool = create_medical_rag_system()
    
    # Test question
    original_question = "What medicines is the patient taking?"
    
    print(f"\n� Original Question: '{original_question}'")
    
    # Step 1: Rewrite question for better retrieval
    rewriter_state = {"messages": [HumanMessage(content=original_question)]}
    rewritten_state = rewrite_question(rewriter_state)
    improved_question = rewritten_state["messages"][0]["content"]
    
    print(f"\n\n🔄 Improved Question: '{improved_question}'")
    
    # Step 2: Retrieve relevant documents
    docs = retriever.invoke(improved_question)
    print(f"\n\n📊 Retrieved {len(docs)} document chunks")
    
    # Combine all retrieved documents for context
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No documents found"
    
    print(f"\n\n📄 Combined Context: {context[:200]}...")
    
    # Step 3: Grade document relevance
    grader_state = {
        "messages": [
            HumanMessage(content=improved_question),
            AIMessage(content=context)
        ]
    }
    decision = grade_documents(grader_state)
    print(f"\n\n⚖️ Grader Decision: {decision}")
    
    # Step 4: Generate answer if documents are relevant
    if decision == "generate_answer":
        answer_state = generate_answer(grader_state)
        medical_answer = answer_state["messages"][0].content
        print(f"\n\n🏥 Medical Answer: {medical_answer}")
    else:
        print("\n\n📝 Documents not relevant - would rewrite question again")

def main():
    """Main function - run the medical RAG workflow test."""
    test_medical_workflow()

if __name__ == "__main__":
    main()
