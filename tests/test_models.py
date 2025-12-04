from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import CompletionRequest, Message, Role


def test_completion_request_valid():
    req = CompletionRequest(
        messages=[
            Message(role=Role.system, content="You are a test assistant."),
            Message(role=Role.user, content="Hello!"),
        ],
        max_tokens=100,
        temperature=0.5,
        top_p=0.9,
    )

    assert req.max_tokens == 100
    assert req.messages[1].role == Role.user


def test_completion_request_requires_user_message():
    with pytest.raises(ValidationError):
        CompletionRequest(
            messages=[Message(role=Role.system, content="System only.")],
        )


def test_completion_request_invalid_max_tokens():
    with pytest.raises(ValidationError):
        CompletionRequest(
            messages=[Message(role=Role.user, content="Hi")],
            max_tokens=0,
        )


