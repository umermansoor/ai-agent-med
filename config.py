"""
Shared configuration for the Medical RAG system.
Loads environment variables once and makes them available to all modules.
"""

from dotenv import load_dotenv
import os

# Load environment variables once
load_dotenv()

# Export commonly used environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please set it in your .env file."
    )

print("✅ Environment variables loaded successfully")
