# Simple Medical RAG System

This is a very simple RAG system for querying patient medical data. This is an example to show limitations of RAG system in complex domains like healthcare.

The system answers medical questions based on raw patient data stored in markdown files. The raw data includes intake forms, lab results, imaging reports, genetic data, and medication lists.

## Setup

1. **Install Dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API Key**
   ```bash
   cp .env.template .env
   # Edit .env and add your OpenAI API and other keys
   ```

3. **Run the System**
   ```bash
   python medical_agent.py
   ```

## Data Structure

The system expects medical data in the `data/` directory with the following structure:
```
data/
  patient_id/
    intake.md
    medications.md
    genetics/
      genetics.md
    imaging/
      *.md
    labs/
      *.md
```