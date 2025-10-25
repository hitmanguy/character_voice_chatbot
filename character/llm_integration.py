import os
import google.generativeai as genai
import logging
import time
from dotenv import load_dotenv  # <-- NEW: Import the library

load_dotenv()

# ============================================================
# 🔌 LLM API INTEGRATION (WITH GOOGLE GEMINI)
# ============================================================
# This module handles all communication with the external Gemini LLM.
# It requires the GOOGLE_API_KEY environment variable to be set.

# --- Initialize the Gemini Client ---
try:
    # 1. Get the API key from environment variables for security
    #os.environ["GOOGLE_API_KEY"] = "AIzaSyDlEVZAra1qsUT6RyT8DWko2O1d5TW8Wns"
    api_key = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)

    # 2. Configure safety settings. For a character AI, we need to be less restrictive
    # to allow for in-character "arrogance" or "wit" without being blocked.
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    # 3. Initialize the generative model
    model = genai.GenerativeModel(
        'gemini-2.5-flash-lite',  # A fast and cost-effective model great for these tasks
        safety_settings=safety_settings
    )
    logging.info("Gemini client initialized successfully.")

except KeyError:
    logging.error("FATAL: GOOGLE_API_KEY environment variable not set.")
    logging.error("Please set your API key to use the LLM-powered services.")
    model = None
except Exception as e:
    logging.error(f"Failed to initialize Gemini client: {e}")
    model = None

def generate_llm_response(prompt: str) -> str:
    """
    Calls the Google Gemini API to generate a response.
    Includes robust error handling and automatic retries for reliability.
    """
    if not model:
        raise ConnectionError("Gemini model not initialized. Check your API key and setup.")
    
    retries = 3
    for attempt in range(retries):
        try:
            # Generate content using the initialized model
            response = model.generate_content(prompt)
            
            # Check if the response was blocked by safety filters
            if not response.parts:
                logging.warning(f"Gemini response was blocked. Prompt: '{prompt[:100]}...'. Reason: {response.prompt_feedback}")
                # Return a safe, generic fallback response
                return "I'd rather not talk about that. Let's move on."

            return response.text.strip()

        except Exception as e:
            logging.error(f"Gemini API call failed on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                # Use exponential backoff for retries to avoid overwhelming the API
                time.sleep(2 ** attempt)
            else:
                logging.error("All retries failed for Gemini API call.")
                return ""  # Return an empty string if all attempts fail