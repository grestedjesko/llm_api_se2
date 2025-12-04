from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str = Field(..., min_length=1)


class CompletionRequest(BaseModel):
    model: Optional[str] = Field(
        default=None,
    )
    messages: List[Message] = Field(
        ...,
        min_length=1,
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
    )

    @field_validator("messages")
    @classmethod
    def validate_messages_have_user(cls, v: List[Message]) -> List[Message]:
        if not any(msg.role == Role.user for msg in v):
            raise ValueError("Request must contain at least one 'user' message")
        return v


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    text: str
    model: str
    usage: Usage
    latency_ms: Optional[float] = Field(default=None)


