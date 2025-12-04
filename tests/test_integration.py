from __future__ import annotations

import sys
from pathlib import Path

import pytest

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_generation():
    from local_llm import LocalLLM
    from config import settings
    
    llm = LocalLLM(model_name=settings.model_name)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, World!' and nothing else."},
    ]
    
    result = await llm.generate_chat(
        messages=messages,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9,
    )
    
    assert "text" in result
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0
    
    assert "prompt_tokens" in result
    assert result["prompt_tokens"] > 0
    
    assert "completion_tokens" in result
    assert result["completion_tokens"] > 0
    
    assert "total_tokens" in result
    assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
    
    assert "latency" in result
    assert result["latency"] > 0
    
    print(f"\n✓ Generation successful!")
    print(f"  Prompt: {messages[-1]['content']}")
    print(f"  Response: {result['text']}")
    print(f"  Tokens: {result['total_tokens']} ({result['prompt_tokens']} + {result['completion_tokens']})")
    print(f"  Latency: {result['latency']:.2f}s")


@pytest.mark.integration
def test_api_endpoint_with_real_model():
    from fastapi.testclient import TestClient
    from api import app
    app.dependency_overrides.clear()
    
    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that responds briefly."},
                {"role": "user", "content": "Count from 1 to 3."},
            ],
            "max_tokens": 30,
            "temperature": 0.5,
            "top_p": 0.9,
        }
        
        resp = client.post("/completions/create", json=payload)
        assert resp.status_code == 200
        
        data = resp.json()
        
        assert "text" in data
        assert isinstance(data["text"], str)
        assert len(data["text"]) > 0
        
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] > 0
        
        assert "latency_ms" in data
        assert data["latency_ms"] > 0
        
        assert "model" in data
        
        print(f"\n✓ API endpoint test successful!")
        print(f"  Response: {data['text']}")
        print(f"  Tokens: {data['usage']['total_tokens']}")
        print(f"  Latency: {data['latency_ms']:.2f}ms")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_generations():
    from local_llm import LocalLLM
    from config import settings
    
    llm = LocalLLM(model_name=settings.model_name)
    
    prompts = [
        "What is 2+2?",
        "Name a color.",
        "Say yes or no.",
    ]
    
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be brief."},
            {"role": "user", "content": prompt},
        ]
        
        result = await llm.generate_chat(
            messages=messages,
            max_new_tokens=15,
            temperature=0.7,
        )
        
        assert result["text"]
        assert result["completion_tokens"] > 0
        print(f"\n  Q: {prompt}")
        print(f"  A: {result['text']}")
    
    print(f"\n✓ All {len(prompts)} generations completed successfully!")

