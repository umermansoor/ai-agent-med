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

# Import the grader
from grader import grade_documents, GradeDocuments

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


if __name__ == "__main__":
    main()
