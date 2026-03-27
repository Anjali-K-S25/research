import os
import logging
from typing import Optional
from dotenv import load_dotenv

from google import genai
from google.genai import types
from groq import Groq

load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(level=logging.INFO)

# ----------------------------
# CLIENTS
# ----------------------------
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


# ----------------------------
# PROMPT BUILDER
# ----------------------------
def build_prompt(context: str, query: str) -> str:
    return f"""
You are a strict research assistant.

Use ONLY the provided context to answer the question.

Rules:
1. Do NOT use external knowledge.
2. If the answer is present, answer clearly.
3. If the answer is NOT present, respond exactly:

Answer: Not found in the retrieved papers.
Research Paper: None

4. You may mention at most 3 research papers.

Response format:

Answer:
<answer>

Research Paper:
<paper title 1>, <paper title 2>, <paper title 3>

Context:
{context}

Question:
{query}
"""


# ----------------------------
# GEMINI CALL
# ----------------------------
def call_gemini(prompt: str) -> Optional[str]:
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


# ----------------------------
# GROQ CALL
# ----------------------------
def call_groq(prompt: str) -> Optional[str]:
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def ask_gemini(context: str, query: str) -> str:
    prompt = build_prompt(context, query)

    # Try Gemini first
    try:
        logging.info("Trying Gemini")

        response = call_gemini(prompt)
        if response:
            return response

    except Exception as e:
        logging.warning(f"Gemini failed: {e}")

    # Fallback to Groq
    try:
        logging.info("Switching to Groq")

        response = call_groq(prompt)
        if response:
            return response

    except Exception as e:
        logging.warning(f"Groq failed: {e}")

    return "Answer:\nNot found due to API failure.\n\nResearch Paper:\nNone"


