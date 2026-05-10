"""
Document management endpoints.

POST /documents/upload   — upload + ingest a PDF
GET  /documents/         — list user's documents (with analytics from materialized view)
GET  /documents/{id}     — single document detail
DELETE /documents/{id}   — delete document + all chunks (cascade)
POST /documents/{id}/summarise — full-document summary
"""

import os
import tempfile

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.db.client import supabase
from app.generation.gemini import summarise
from app.ingestion.pipeline import ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])

_STORAGE_BUCKET = "documents"
_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(401, "X-User-Id header required")
    return x_user_id


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_user_id: str | None = Header(default=None),
):
    user_id = _require_user(x_user_id)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    content = await file.read()
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(413, "File exceeds 50 MB limit")

    # Upload to Supabase Storage under user's namespace
    storage_path = f"{user_id}/{file.filename}"
    supabase.storage.from_(_STORAGE_BUCKET).upload(
        storage_path, content, {"content-type": "application/pdf"}
    )
    file_url = supabase.storage.from_(_STORAGE_BUCKET).get_public_url(storage_path)

    # Write to temp file so PyMuPDF can open it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await ingest_document(tmp_path, user_id, file.filename, file_url)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    return JSONResponse(result, status_code=201)


@router.get("/")
def list_documents(x_user_id: str | None = Header(default=None)):
    user_id = _require_user(x_user_id)
    result = (
        supabase.table("documents")
        .select("id, filename, doc_type, chunk_count, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


@router.get("/{document_id}")
def get_document(document_id: str, x_user_id: str | None = Header(default=None)):
    user_id = _require_user(x_user_id)
    result = (
        supabase.table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "Document not found")
    return result.data


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, x_user_id: str | None = Header(default=None)):
    user_id = _require_user(x_user_id)
    supabase.table("documents").delete().eq("id", document_id).eq(
        "user_id", user_id
    ).execute()


@router.post("/{document_id}/summarise")
def summarise_document(document_id: str, x_user_id: str | None = Header(default=None)):
    user_id = _require_user(x_user_id)

    # Fetch doc type for prompt routing
    doc = (
        supabase.table("documents")
        .select("doc_type")
        .eq("id", document_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not doc.data:
        raise HTTPException(404, "Document not found")

    # Fetch all chunks ordered by chunk_index
    chunks_result = (
        supabase.table("chunks")
        .select("content, metadata")
        .eq("document_id", document_id)
        .eq("user_id", user_id)
        .order("metadata->>chunk_index")
        .execute()
    )
    chunks = chunks_result.data
    if not chunks:
        raise HTTPException(422, "Document has no processable chunks")

    summary = summarise(chunks, doc.data.get("doc_type", "general"))
    return {"summary": summary}
