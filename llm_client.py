"""
Shared LLM client config — OpenAI SDK pointed at Groq's OpenAI-compatible API.

Both the extraction agent (structured outputs) and the summary agent use this,
so the model + endpoint + key are configured in one place.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # read GROQ_API_KEY from .env

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(base_url=GROQ_BASE_URL, api_key=api_key or os.environ.get("GROQ_API_KEY"))
