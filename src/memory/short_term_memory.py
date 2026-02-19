"""
short_term_memory.py
---------------------
Handles fetching and formatting short-term conversation history
from the memory service (e.g., a Rasa-compatible store on port 8000).
"""

import logging
import requests

logger = logging.getLogger(__name__)

MEMORY_SERVICE_URL = "http://localhost:8000/fetch_messages"
REQUEST_TIMEOUT    = 10  # seconds


def fetch_conversation_history(sender_id: str) -> list[dict]:
    """
    Fetch recent conversation messages for a given sender from the memory service.

    Args:
        sender_id: Unique identifier for the user/session.

    Returns:
        List of message dicts with 'sender' and 'text' keys,
        or an empty list if the service is unreachable.
    """
    try:
        response = requests.post(
            MEMORY_SERVICE_URL,
            json={"sender_id": sender_id},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        messages = response.json().get("messages", [])
        logger.info(f"💬 Retrieved {len(messages)} messages for sender '{sender_id}'.")
        return messages

    except requests.RequestException as e:
        logger.warning(f"⚠️ Could not reach memory service: {e}. Proceeding without history.")
        return []


def format_history_for_prompt(messages: list[dict]) -> str:
    """
    Convert a list of message dicts into a plain-text conversation string
    suitable for injection into an LLM prompt.

    Args:
        messages: List of {'sender': 'user'|'bot', 'text': str} dicts.

    Returns:
        Formatted string like "User: ...\nAssistant: ..."
    """
    if not messages:
        return "No prior conversation."

    lines = []
    for msg in messages:
        role = "User" if msg.get("sender") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('text', '')}")
    return "\n".join(lines)
