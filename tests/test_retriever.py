"""
test_retriever.py
-----------------
Unit tests for the retrieval pipeline.
Run with: pytest tests/
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Test: format_history_for_prompt ──────────────────────────────────────────
def test_format_history_empty():
    from src.memory.short_term_memory import format_history_for_prompt
    result = format_history_for_prompt([])
    assert result == "No prior conversation."


def test_format_history_with_messages():
    from src.memory.short_term_memory import format_history_for_prompt
    messages = [
        {"sender": "user", "text": "Hello"},
        {"sender": "bot",  "text": "Hi there!"},
    ]
    result = format_history_for_prompt(messages)
    assert "User: Hello" in result
    assert "Assistant: Hi there!" in result


# ── Test: fetch_conversation_history (mocked) ────────────────────────────────
@patch("src.memory.short_term_memory.requests.post")
def test_fetch_history_success(mock_post):
    from src.memory.short_term_memory import fetch_conversation_history
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"messages": [{"sender": "user", "text": "Hi"}]}
    )
    result = fetch_conversation_history("test_user")
    assert len(result) == 1
    assert result[0]["text"] == "Hi"


@patch("src.memory.short_term_memory.requests.post", side_effect=Exception("Connection refused"))
def test_fetch_history_failure_returns_empty(mock_post):
    from src.memory.short_term_memory import fetch_conversation_history
    result = fetch_conversation_history("test_user")
    assert result == []
