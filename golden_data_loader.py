"""
Utility functions for loading golden question data.
Centralizes JSONL file loading to avoid code duplication.
"""

import json
import os
from typing import Dict, Any, Optional


def load_golden_questions_raw(patient_id: str = "drapoel") -> Dict[str, Any]:
    """
    Load raw golden questions data from JSONL file.
    Returns a dictionary with question_id as keys and full question data as values.
    """
    golden_file = f"golden_data/{patient_id}/golden.jsonl"
    if not os.path.exists(golden_file):
        raise FileNotFoundError(f"Golden questions file not found: {golden_file}")
    
    questions = {}
    with open(golden_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                question_data = json.loads(line)
                questions[question_data['id']] = question_data
    
    return questions


def load_golden_answers_formatted(patient_id: str = "drapoel") -> Dict[str, str]:
    """
    Load golden answers formatted for judge evaluation.
    Returns a dictionary with question_id as keys and formatted golden answers as values.
    """
    try:
        questions_data = load_golden_questions_raw(patient_id)
        golden_answers = {}
        
        for question_id, question_data in questions_data.items():
            if "golden_answer" in question_data:
                # Join content array into a single string
                golden_content = "\n".join(question_data["golden_answer"]["content"])
                ideal_context = question_data["golden_answer"].get("ideal_context", [])
                
                # Format like the old structure with IDEAL CONTEXT section
                if ideal_context:
                    golden_content += "\n\n-------------------------------------------\n"
                    golden_content += "IDEAL CONTEXT (must be present to generate a good answer):\n"
                    for context_item in ideal_context:
                        golden_content += f"- {context_item}\n"
                
                golden_answers[question_id] = golden_content
        
        return golden_answers
    except Exception as e:
        print(f"Error loading golden answers: {e}")
        return {}


def get_question_by_id(question_id: str, patient_id: str = "drapoel") -> Optional[Dict[str, Any]]:
    """Get a specific question by ID."""
    questions = load_golden_questions_raw(patient_id)
    return questions.get(question_id)
