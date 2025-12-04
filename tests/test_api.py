from __future__ import annotations

from typing import Any, AsyncIterator, Dict

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api import app, get_llm


class MockLLM:
    async def generate_chat(
        self,
        messages,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        return {
            "text": "mock-response",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "latency": 0.01,
        }


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app.dependency_overrides[get_llm] = lambda: MockLLM()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_ok(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_name" in data


@pytest.mark.asyncio
async def test_create_completion_success(client: AsyncClient):
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"},
        ],
        "max_tokens": 16,
        "temperature": 0.5,
        "top_p": 0.9,
    }
    resp = await client.post("/completions/create", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "mock-response"
    assert data["usage"]["prompt_tokens"] == 10
    assert data["usage"]["completion_tokens"] == 5
    assert data["usage"]["total_tokens"] == 15
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_create_completion_validation_error(client: AsyncClient):
    payload = {
        "messages": [
            {"role": "system", "content": "Only system message."},
        ],
    }
    resp = await client.post("/completions/create", json=payload)
    assert resp.status_code == 422


