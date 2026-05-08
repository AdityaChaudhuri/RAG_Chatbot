"""
Ingestion orchestrator.

Full pipeline for a single PDF upload:
  parse → NER + classify (parallel) → semantic chunk → embed → store
"""

import asyncio
from pathlib import Path

from app.db.client import supabase
from app.ingestion.classifier import classify_document
from app.ingestion.chunker import semantic_chunk
from app.ingestion.embedder import embed_documents
from app.ingestion.ner import extract_entities
from app.ingestion.parser import full_text, parse_pdf

_CHUNK_INSERT_BATCH = 100


async def ingest_document(
    file_path: str | Path,
    user_id: str,
    filename: str,
    file_url: str,
) -> dict:
    """
    Parse, enrich, embed, and store a PDF document.

    Returns:
        {document_id, doc_type, entity_tags, chunk_count}
    """
    pages = parse_pdf(file_path)
    if not pages:
        raise ValueError(f"No extractable text found in {filename}")

    text = full_text(pages)

    # NER and classification are both CPU-bound — run them concurrently.
    loop = asyncio.get_event_loop()
    entity_tags, doc_type = await asyncio.gather(
        loop.run_in_executor(None, extract_entities, text),
        loop.run_in_executor(None, classify_document, text),
    )

    # Insert the document row first so chunk foreign keys have a target.
    # chunk_count starts at 0 and is incremented by trigger on each chunk insert.
    doc_result = (
        supabase.table("documents")
        .insert({
            "user_id": user_id,
            "filename": filename,
            "file_url": file_url,
            "doc_type": doc_type,
            "entity_tags": entity_tags,
        })
        .execute()
    )
    document_id: str = doc_result.data[0]["id"]

    chunks = semantic_chunk(pages)
    texts = [c["content"] for c in chunks]
    embeddings = await loop.run_in_executor(None, embed_documents, texts)

    records = [
        {
            "document_id": document_id,
            "user_id": user_id,
            "content": chunk["content"],
            "embedding": embedding,
            "token_count": chunk["token_count"],
            "metadata": {
                "page_nums": chunk["page_nums"],
                "chunk_index": chunk["chunk_index"],
            },
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    # Batch inserts to stay within Supabase's request body size limit.
    for i in range(0, len(records), _CHUNK_INSERT_BATCH):
        supabase.table("chunks").insert(records[i : i + _CHUNK_INSERT_BATCH]).execute()

    return {
        "document_id": document_id,
        "doc_type": doc_type,
        "entity_tags": entity_tags,
        "chunk_count": len(chunks),
    }
