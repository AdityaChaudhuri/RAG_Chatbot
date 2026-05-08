from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, chat

app = FastAPI(
    title="Mr.Summarizer API",
    description="Production RAG pipeline — hybrid retrieval, ML re-ranking, Gemini generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://mr-summarizer.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "service": "mr-summarizer"}
