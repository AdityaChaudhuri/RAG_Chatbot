"""
Prompt templates.

Each document type gets a tailored system instruction and context framing.
Routing by doc_type (set at ingest time by the classifier) ensures Claude
applies domain-appropriate caution and terminology.
"""

_SYSTEM_BASE = (
    "You are Mr.Summarizer, an expert document analyst. "
    "Answer questions based strictly on the provided document context. "
    "Cite the specific pages or sections that support your answer. "
    "If the context does not contain sufficient information, say so clearly — "
    "do not speculate or use outside knowledge."
)

_DOC_TYPE_INSTRUCTIONS: dict[str, str] = {
    "legal": (
        "This is a legal document. Be precise about obligations, rights, and liabilities. "
        "Use the exact terminology from the document. Flag any ambiguous clauses."
    ),
    "academic": (
        "This is an academic paper. Reference methodologies, statistical findings, "
        "and citations accurately. Distinguish between the authors' claims and their evidence."
    ),
    "financial": (
        "This is a financial document. Be precise with figures, dates, and financial "
        "terminology. Do not round or paraphrase numbers."
    ),
    "technical": (
        "This is a technical document. Focus on specifications, APIs, and implementation "
        "details. Preserve exact identifiers, version numbers, and configuration values."
    ),
    "general": (
        "Answer the question based on the document content provided."
    ),
}

_SUMMARISE_INSTRUCTION = (
    "Produce a comprehensive, well-structured summary of the document. "
    "Include: main topic, key arguments or findings, important entities mentioned, "
    "and any conclusions or recommendations. Use clear headings."
)


def system_prompt() -> str:
    return _SYSTEM_BASE


def build_rag_prompt(
    query: str,
    chunks: list[dict],
    doc_type: str = "general",
) -> str:
    """
    Construct the user-turn prompt for a RAG query.

    Chunks are formatted with their page provenance so Claude can cite them.
    The doc-type instruction focuses Claude on domain-appropriate behaviour.
    """
    instruction = _DOC_TYPE_INSTRUCTIONS.get(doc_type, _DOC_TYPE_INSTRUCTIONS["general"])

    context_blocks = "\n\n---\n\n".join(
        "[Page {pages}]\n{content}".format(
            pages=", ".join(str(p) for p in chunk.get("metadata", {}).get("page_nums", ["?"])),
            content=chunk["content"],
        )
        for chunk in chunks
    )

    return (
        f"{instruction}\n\n"
        f"Document context:\n\n{context_blocks}\n\n"
        f"Question: {query}\n\n"
        "Answer based only on the document context above. "
        "Reference specific pages where relevant."
    )


def build_summarise_prompt(chunks: list[dict], doc_type: str = "general") -> str:
    instruction = _DOC_TYPE_INSTRUCTIONS.get(doc_type, _DOC_TYPE_INSTRUCTIONS["general"])
    full_content = "\n\n".join(c["content"] for c in chunks)
    return (
        f"{instruction}\n\n"
        f"{_SUMMARISE_INSTRUCTION}\n\n"
        f"Document:\n\n{full_content}"
    )
