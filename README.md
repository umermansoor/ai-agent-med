# ai-agent-med

A Retrieval-Augmented Generation (RAG) system for querying patient medical data using Langgraph. This system provides an interactive shell where you can ask questions about patient health information stored in markdown files.

## Features

- **RAG System**: Uses vector embeddings to retrieve relevant medical information
- **Interactive Shell**: Command-line interface for natural language queries
- **Medical Data Processing**: Automatically loads and processes patient medical documents
- **Intelligent Retrieval**: Finds relevant information across multiple document types (intake forms, lab results, medications, imaging, etc.)

## Setup

1. **Install Dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   pip install langgraph langchain langchain-community langchain-openai langchain-chroma chromadb tiktoken python-dotenv
   ```

2. **Configure OpenAI API Key**
   ```bash
   cp .env.template .env
   # Edit .env and add your OpenAI API key
   ```

3. **Run the System**
   ```bash
   python medical_rag_agent.py
   ```

## Usage

Once running, you can ask questions about patient medical data:

- "What medications is the patient taking?"
- "What are the patient's recent lab results?"
- "Tell me about the patient's cholesterol levels"
- "What imaging studies were performed?"
- "What are the patient's primary health concerns?"
- "Summarize the patient's genetic information"

Type `help` for more example questions or `quit` to exit.

## Data Structure

The system expects medical data in the `data/` directory with the following structure:
```
data/
  patient_name/
    intake.md
    medications.md
    genetics/
      genetics.md
    imaging/
      *.md
    labs/
      *.md
```

## Implementation Details

This RAG system implements the following components:

1. **Document Loading**: Recursively loads all markdown files from the data directory
2. **Text Splitting**: Chunks documents for optimal retrieval performance
3. **Vector Store**: Uses Chroma DB with OpenAI embeddings for semantic search
4. **Retriever Tool**: Langgraph tool that searches relevant medical information
5. **Agent**: Langgraph agent that combines LLM reasoning with retrieval capabilities

The implementation follows tutorial steps up to Step 2: "Create a retriever tool".