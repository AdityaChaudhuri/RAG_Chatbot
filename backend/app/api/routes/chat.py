"""
Chat endpoints.

POST /chat/sessions                         — create a session
GET  /chat/sessions                         — list user's sessions
GET  /chat/sessions/{id}/messages           — fetch message history
POST /chat/sessions/{id}/messages           — send a message (SSE stream)
"""

import json
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.db.client import supabase
from app.generation.claude import stream_answer
from app.retrieval.compressor import compress_chunks
from app.retrieval.multi_query import multi_query_retrieve
from app.retrieval.reranker import rerank

router = APIRouter(prefix="/chat", tags=["chat"])


def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(401, "X-User-Id header required")
    return x_user_id


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@router.post("/sessions", status_code=201)
def create_session(
    document_id: str | None = None,
    x_user_id: str | None = Header(default=None),
):
    user_id = _require_user(x_user_id)
    result = (
        supabase.table("chat_sessions")
        .insert({"user_id": user_id, "document_id": document_id})
        .execute()
    )
    return result.data[0]


@router.get("/sessions")
def list_sessions(x_user_id: str | None = Header(default=None)):
    user_id = _require_user(x_user_id)
    result = (
        supabase.table("chat_sessions")
        .select("id, document_id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data


@router.get("/sessions/{session_id}/messages")
def get_messages(
    session_id: str,
    x_user_id: str | None = Header(default=None),
):
    user_id = _require_user(x_user_id)
    result = (
        supabase.table("chat_history")
        .select("id, role, content, source_chunks, retrieval_score, created_at")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("created_at")
        .execute()
    )
    return result.data


# ---------------------------------------------------------------------------
# Message send (streaming)
# ---------------------------------------------------------------------------

class MessageRequest(BaseModel):
    query: str
    document_id: str | None = None


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    req: MessageRequest,
    x_user_id: str | None = Header(default=None),
):
    user_id = _require_user(x_user_id)
    start_ms = int(time.time() * 1000)

    # ---- Validate session belongs to user ----------------------------------
    session = (
        supabase.table("chat_sessions")
        .select("document_id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not session.data:
        raise HTTPException(404, "Session not found")

    doc_id = req.document_id or session.data.get("document_id")

    # ---- Fetch doc_type for prompt routing ---------------------------------
    doc_type = "general"
    if doc_id:
        doc = (
            supabase.table("documents")
            .select("doc_type")
            .eq("id", doc_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        doc_type = (doc.data or {}).get("doc_type", "general")

    # ---- Fetch recent conversation history ---------------------------------
    history_result = (
        supabase.rpc("get_session_context", {"p_session_id": session_id, "p_limit": 20})
        .execute()
    )
    history = [{"role": r["role"], "content": r["content"]} for r in (history_result.data or [])]

    # ---- Retrieval pipeline ------------------------------------------------
    pool = multi_query_retrieve(req.query, user_id, doc_id)
    top = rerank(req.query, pool)
    context = compress_chunks(req.query, top)

    # ---- Persist user message ---------------------------------------------
    supabase.table("chat_history").insert(
        {
            "session_id": session_id,
            "user_id": user_id,
            "role": "user",
            "content": req.query,
        }
    ).execute()

    # ---- Stream assistant response ----------------------------------------
    async def event_stream() -> AsyncGenerator[str, None]:
        tokens: list[str] = []

        for token in stream_answer(req.query, context, history, doc_type):
            tokens.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        full_response = "".join(tokens)
        latency_ms = int(time.time() * 1000) - start_ms
        top_score = top[0].get("rerank_score", 0.0) if top else 0.0
        cited_ids = [c["chunk_id"] for c in context]

        supabase.table("chat_history").insert(
            {
                "session_id": session_id,
                "user_id": user_id,
                "role": "assistant",
                "content": full_response,
                "source_chunks": cited_ids,
                "retrieval_score": top_score,
                "latency_ms": latency_ms,
            }
        ).execute()

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
