# llm_engine.py
"""
================================================================================
LLM ENGINE
================================================================================
Handles interaction with local or cloud LLMs for summaries & Q&A
"""

import json
import requests

import config


def _call_local_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Call local LLM via Ollama-style API.
    Adjust if your local stack returns a different JSON schema.
    """
    payload = {
        "model": config.LOCAL_LLM_MODEL,
        "prompt": prompt,
        "temperature": temperature,
    }
    try:
        resp = requests.post(config.LOCAL_LLM_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"model": "...", "created_at": "...", "response": "...", ...}
        if "response" in data:
            return data["response"].strip()
        # Fallback
        return json.dumps(data)
    except Exception as e:
        return f"[LLM error] {e}"


def generate_summary(article_text: str) -> str:
    prompt = (
        "You are an institutional-grade financial news analyst. "
        "Read the article below and produce a concise summary in 3–5 bullet points. "
        "Focus on market impact, key drivers, and risk factors. "
        "Do NOT provide financial advice.\n\n"
        f"ARTICLE:\n{article_text}\n\n"
        "Now give the bullet-point summary:"
    )
    return _call_local_llm(prompt)


def answer_question(article_text: str, question: str) -> str:
    prompt = (
        "You are an institutional financial research assistant. "
        "Use ONLY the information in the article below to answer the user's question. "
        "If the answer is not clearly supported by the article, say that it is not specified.\n\n"
        f"ARTICLE:\n{article_text}\n\n"
        f"QUESTION: {question}\n\n"
        "Provide a clear, concise answer in 3–6 sentences, without financial advice:"
    )
    return _call_local_llm(prompt)
