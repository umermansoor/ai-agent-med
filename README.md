# ai-agent-med

A Retrieval-Augmented Generation (RAG) system for querying patient medical data. This is an example to show limitations of RAG system in complex domains like healthcare.

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
   python medical_rag_components.py
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

## Example Output

### Rewritten Questions
```sh
((.venv) ) (base) umermansoor@Umers-MacBook ai-agent-med % python medical_rag_agent.py
✅ Environment variables loaded successfully

� Original Question: 'does the patient have diabetes?'
🔄 Improved Question: 'Based on the patient's laboratory results, specifically the hemoglobin A1C levels and fasting blood glucose measurements, is there evidence to support a diagnosis of diabetes mellitus? Additionally, are there any current medications or clinical findings that indicate management of diabetes?'

� Original Question: 'list the patient's current medications'
🔄 Improved Question: 'Could you provide a detailed list of the patient's current pharmacological treatments, including all prescribed medications and over-the-counter supplements, along with their respective dosages and indications?'

� Original Question: 'the person reported feeling fatigued and weak. what could be the cause?'
🔄 Improved Question: 'Considering the patient's reported symptoms of fatigue and weakness, what laboratory abnormalities, such as anemia, electrolyte imbalances, or thyroid dysfunction, could be contributing to these clinical findings? Additionally, are there any current medications or supplements that might be causing these symptoms as side effects?'

� Original Question: 'what's the health status of the patient?'
🔄 Improved Question: 'Could you provide a comprehensive overview of the patient's current health status, including recent laboratory results, current medications and dosages, imaging study findings, and any relevant genetic information?'
```

### Example Answer
```sh
((.venv) ) (base) umermansoor@Umers-MacBook ai-agent-med % python medical_rag_agent.py
/opt/homebrew/Cellar/python@3.12/3.12.10_1/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python: can't open file '/Users/umermansoor/Documents/GitHub/ai-agent-med/medical_rag_agent.py': [Errno 2] No such file or directory
((.venv) ) (base) umermansoor@Umers-MacBook-Pro-2 ai-agent-med % python medical_rag_components.py 
✅ Environment variables loaded successfully
📚 Loaded 12 medical documents
📝 Split into 59 chunks
🔍 Creating embeddings and vector store...
📊 Retriever configured to return top 3 most relevant chunks
✅ Retrieval tool ready

� Original Question: 'what prescription medications is the patient taking?'


🔄 Improved Question: 'Could you provide a detailed list of the patient's current prescription medications, including the drug names, dosages, and indications for use?'


📊 Retrieved 3 document chunks


📄 Combined Context: ## Prescription Medications

---

## Medications (Current)  
- Levothyroxine 75 mcg once daily  
- Atorvastatin 20 mg nightly  
- Aspirin 81 mg daily  
- Vitamin D3 2000 IU daily  
- Calcium Citrate 5...


⚖️ Grader Decision: generate_answer


🏥 Medical Answer: The patient's current prescription medications include Levothyroxine 75 mcg once daily for thyroid hormone replacement, Atorvastatin 20 mg nightly for hyperlipidemia management, and Aspirin 81 mg daily for cardiovascular prophylaxis. Additionally, the patient takes Vitamin D3 2000 IU daily and Calcium Citrate 500 mg twice daily for bone health, Omega-3 Fish Oil 1000 mg twice daily for cardiovascular support, and a daily multivitamin for general health maintenance.
```