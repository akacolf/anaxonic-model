"""
serve.py — HTTP server for anaxonic-model (John Sakura 1.0)

Exposes an OpenAI-compatible /v1/chat/completions endpoint so the John
frontend (john.quantumclub.sh) can call it directly via the Vercel AI SDK.

Usage:
    python -m anaxonic.serve                          # defaults
    python -m anaxonic.serve --host 0.0.0.0 --port 8000
    python -m anaxonic.serve --checkpoint runs/hf_chat_frontier_course_balanced/best.pt

Deploy:
    Docker: docker build -t anaxonic-serve . && docker run -p 8000:8000 anaxonic-serve serve
    Railway / Render / Fly.io: set start command to `python -m anaxonic.serve`
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing serving dependencies. Install them with:\n"
        '  pip install "anaxonic-model[serve]"'
    ) from e

from anaxonic.chat import ChatSession, DEFAULT_CHECKPOINT, DEFAULT_MEMORY_DATA, DEFAULT_LEARN_DATA

MODEL_NAME = "john-sakura-1.0"

JOHN_SYSTEM = (
    "You are John, an AI assistant built by Anaxonic — a company building AI and blockchain "
    "products for onchain developers. You are helpful, concise, and technical. "
    "Answer directly and accurately. Say when you are unsure rather than guessing."
)

# ---------------------------------------------------------------------------
# Pydantic models (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Anaxonic Model API",
    description="OpenAI-compatible inference endpoint for John Sakura 1.0",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://john.quantumclub.sh", "http://localhost:3000"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Global session — loaded once at startup
_session: ChatSession | None = None


def get_session() -> ChatSession:
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _session


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_chunk(content: str, finish_reason: str | None = None) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def _stream_response(text: str) -> AsyncIterator[str]:
    """Simulate token streaming by yielding word-by-word."""
    words = text.split(" ")
    for i, word in enumerate(words):
        token = word if i == 0 else " " + word
        yield _sse_chunk(token)
        # small yield to let the event loop breathe
        import asyncio
        await asyncio.sleep(0)
    yield _sse_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "loaded": _session is not None,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "anaxonic",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    session = get_session()

    # Extract the last user message
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    prompt = user_messages[-1].content

    # Run inference (synchronous — wrap in thread if needed for async)
    import asyncio
    loop = asyncio.get_event_loop()
    response_text = await loop.run_in_executor(None, session.ask, prompt)

    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if req.stream:
        return StreamingResponse(
            _stream_response(response_text),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(prompt.split()) + len(response_text.split()),
        },
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Serve John Sakura 1.0 over HTTP")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--memory-data", default=DEFAULT_MEMORY_DATA)
    parser.add_argument("--learn-data", default=DEFAULT_LEARN_DATA)
    parser.add_argument(
        "--generate-unknown",
        action="store_true",
        dest="generate_unknown",
        help="Allow neural generation for prompts not found in memory.",
    )
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} from {args.checkpoint} …")
    global _session
    _session = ChatSession(
        checkpoint_path=args.checkpoint,
        system=JOHN_SYSTEM,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        memory_data=None if args.no_memory else args.memory_data,
        learn_data=args.learn_data,
        generate_unknown=args.generate_unknown,
    )
    print(f"✓ {MODEL_NAME} ready — listening on http://{args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
