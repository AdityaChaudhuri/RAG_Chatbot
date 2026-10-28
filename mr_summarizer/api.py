"""FastAPI app.

Run:  uvicorn mr_summarizer.api:app --reload
Then open http://localhost:8000

Endpoints
  GET /ask?q=...          answer streamed as Server-Sent Events
  GET /search?q=...       retrieval only, with each retriever's ranks (for debugging and demos)
  GET /graph/{name}       an entity and its strongest neighbours in the knowledge graph
"""
import json
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from . import llm
from .db import connect
from .graph import normalise
from .retrieval import Hit, retrieve

app = FastAPI(title="Mr. Summarizer")
STATIC = Path(__file__).resolve().parent.parent / "static"


def sse(event: str, data) -> str:
    """Format one Server-Sent Event. Data is JSON so newlines in text stay safe."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def hit_json(i: int, h: Hit) -> dict:
    return {
        "n": i,
        "title": h.title,
        "url": h.url,
        "content": h.content,
        "ranks": h.ranks,
        "rrf": round(h.rrf_score, 5),
        "rerank": None if h.rerank_score is None else round(h.rerank_score, 3),
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/ask")
def ask(q: str = Query(..., min_length=3)):
    def events() -> Iterator[str]:
        r = retrieve(q)
        sources = [hit_json(i, h) for i, h in enumerate(r.hits, start=1)]
        # send sources first so the page can show them while the answer streams
        yield sse("sources", {"entities": r.entities, "sources": sources})
        if not sources:
            yield sse("token", "I couldn't find anything relevant in the indexed news.")
        else:
            try:
                for token in llm.stream_answer(q, sources):
                    yield sse("token", token)
            except Exception as e:  # surface LLM/config errors in the UI instead of a dead stream
                yield sse("error", f"{type(e).__name__}: {e}")
        yield sse("done", None)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/search")
def search(q: str = Query(..., min_length=3), rerank: bool = True):
    r = retrieve(q, use_rerank=rerank)
    return {
        "entities": r.entities,
        "retriever_results": {k: len(v) for k, v in r.ranked_lists.items()},
        "hits": [hit_json(i, h) for i, h in enumerate(r.hits, start=1)],
    }


@app.get("/graph/{name}")
def entity_graph(name: str, limit: int = 15):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT e.id, e.name, e.label, count(ce.chunk_id) AS mentions
               FROM entities e LEFT JOIN chunk_entities ce ON ce.entity_id = e.id
               WHERE e.key = %s GROUP BY e.id ORDER BY mentions DESC LIMIT 1""",
            (normalise(name),),
        )
        node = cur.fetchone()
        if not node:
            raise HTTPException(404, f"No entity called '{name}' in the graph")
        cur.execute(
            """SELECT n.name, n.label, e.weight
               FROM entity_edges e
               JOIN entities n ON n.id = CASE WHEN e.src = %(id)s THEN e.dst ELSE e.src END
               WHERE %(id)s IN (e.src, e.dst)
               ORDER BY e.weight DESC LIMIT %(limit)s""",
            {"id": node[0], "limit": limit},
        )
        neighbours = [{"name": n, "label": l, "shared_chunks": w} for n, l, w in cur.fetchall()]
    return {"name": node[1], "label": node[2], "mentions": node[3], "neighbours": neighbours}
