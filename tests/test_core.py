"""Unit tests for the pure-Python parts. No database or models needed: `pytest`."""
import json

import pytest

from mr_summarizer import graph, text
from mr_summarizer.fusion import rrf
from mr_summarizer.llm import build_user_prompt


# ---------- cleaning and chunking ----------

def test_clean_removes_boilerplate_and_whitespace():
    raw = "Updated 19 minutes ago  Reuters Staff 2 Min Read\nStocks   fell on Tuesday."
    assert text.clean(raw) == "Stocks fell on Tuesday."


def test_split_sentences():
    assert text.split_sentences("Shares rose. Bonds fell! Why? \"Oil\" held.") == [
        "Shares rose.", "Bonds fell!", "Why?", "\"Oil\" held.",
    ]


def test_chunk_respects_size_and_overlaps():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(40)]
    chunks = text.chunk(" ".join(sentences), max_words=50, overlap_words=14)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 50
    # consecutive chunks share their boundary sentences
    for a, b in zip(chunks, chunks[1:]):
        assert a.split(". ")[-1].rstrip(".") in b
    # nothing is lost
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_chunk_short_text_is_one_chunk():
    assert text.chunk("Just one short sentence.") == ["Just one short sentence."]


def test_chunk_long_single_sentence_kept_whole():
    long = " ".join(["word"] * 300) + "."
    assert text.chunk(long, max_words=100) == [long]


# ---------- knowledge graph helpers ----------

@pytest.mark.parametrize("raw,key", [
    ("Goldman Sachs", "goldman sachs"),
    ("the Federal Reserve", "federal reserve"),
    ("Apple's", "apple"),
    ("  Tesla  Inc. ", "tesla inc"),
])
def test_normalise(raw, key):
    assert graph.normalise(raw) == key


def test_keep_filters_labels_and_noise():
    assert graph.keep("goldman sachs", "ORG")
    assert not graph.keep("goldman sachs", "DATE")
    assert not graph.keep("reuters", "ORG")
    assert not graph.keep("x", "ORG")


def test_cooccurrence_edges_counts_shared_chunks():
    edges = graph.cooccurrence_edges([{1, 2, 3}, {2, 1}, {3}])
    assert edges == {(1, 2): 2, (1, 3): 1, (2, 3): 1}


def test_entities_in_query_prefers_longest_match():
    known = {"goldman sachs": [1], "goldman": [2], "federal reserve": [3, 4], "fed": [5]}
    found = graph.entities_in_query("What did Goldman Sachs say about the Federal Reserve?", known)
    assert sorted(found) == [1, 3, 4]


def test_entities_in_query_whole_words_only():
    assert graph.entities_in_query("federation news", {"fed": [5]}) == []


# ---------- reciprocal rank fusion ----------

def test_rrf_rewards_agreement():
    fused = rrf({"keyword": [10, 20, 30], "vector": [20, 40], "graph": [20]}, k=60)
    assert fused[0].chunk_id == 20
    assert fused[0].ranks == {"keyword": 2, "vector": 1, "graph": 1}
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)


def test_rrf_handles_empty_and_duplicates():
    fused = rrf({"keyword": [], "vector": [5, 5, 6]}, k=60)
    assert [f.chunk_id for f in fused] == [5, 6]
    assert fused[0].score == pytest.approx(1 / 61)


# ---------- prompt ----------

def test_prompt_numbers_sources_in_order():
    p = build_user_prompt("Why?", [{"title": "A", "content": "x"}, {"title": "B", "content": "y"}])
    assert p.index("[1] A\nx") < p.index("[2] B\ny")
    assert p.endswith("Question: Why?")


# ---------- server-sent events ----------

def test_sse_format():
    api = pytest.importorskip("mr_summarizer.api")  # skipped if FastAPI is not installed
    out = api.sse("token", "line one\nline two")
    assert out.startswith("event: token\ndata: ")
    assert out.endswith("\n\n")
    assert json.loads(out.split("data: ", 1)[1]) == "line one\nline two"
