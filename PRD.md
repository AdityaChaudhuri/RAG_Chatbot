# Mr.Summarizer — Product Requirements Document
### Document Intelligence Platform

---

## 1. Overview

Mr.Summarizer is a web application that turns any PDF into a queryable knowledge base. Users upload any document — a legal contract, research paper, technical manual, trade regulation, company policy, financial report — and ask plain English questions about it. The system understands not just what each document says, but how the concepts, entities, and rules inside it connect to each other.

The core insight: most documents are not a list of facts, they are a web of relationships. A contract clause references a defined term which references a schedule which references a third-party certification. A research paper cites a methodology which assumes a prior finding which contradicts a newer study. Standard document search finds words. Mr.Summarizer finds connections.

---

## 2. The Problem

People upload PDFs to ChatGPT and ask questions. They get answers that miss context, skip relationships, and hallucinate details not in the document. The model reads text linearly — it cannot traverse the structure of what it just read.

The result: you still have to read the document yourself to trust the answer.

Mr.Summarizer fixes this by building a knowledge graph from every document it processes. Instead of searching for relevant text, it walks the relationships between entities to assemble a grounded, traceable answer.

---

## 3. Who This Is For

**Primary user:** Anyone who regularly reads dense documents and needs to extract specific information quickly — legal teams, researchers, operations staff, consultants, students.

**Secondary user:** Small teams that share and reference the same set of documents and need a shared intelligence layer on top of them.

---

## 4. Goals

- Upload any PDF and immediately ask questions about it
- Answer relational questions that standard search cannot handle
- Surface connections between entities across a document or across multiple documents
- Save every conversation so users can reference past queries
- Be simple enough that anyone can use it without a technical background

## 4.1 Non-Goals

- We are not replacing careful human reading for high-stakes decisions
- We are not connecting to live external data sources in v1 — users upload documents manually
- We are not supporting file types other than PDF in v1
- We are not building real-time collaboration in v1

---

## 5. Core Features

### 5.1 Document Upload and Processing
- Drag and drop any PDF
- The system processes it in the background: extracts text, identifies entities, maps relationships, indexes for search
- User is notified when the document is ready to query

### 5.2 Hybrid Retrieval — Two Modes

**Semantic Search**
Used for straightforward factual questions.
> *"What is the termination notice period in this contract?"*
Finds the most relevant chunks of text and passes them to the model.

**Graph Traversal**
Used for relational questions that require connecting multiple pieces of information.
> *"How does the liability clause connect to the indemnification schedule?"*
The system walks the graph: liability clause → references → indemnification schedule → defines → cap amount → answer.

**Query Router**
Reads the question and automatically decides which mode to use, or whether to combine both.

### 5.3 Document Summarisation
- One-click summary of any uploaded document
- Structured output: what the document covers, key entities, important dates, main requirements or findings, notable relationships

### 5.4 Entity Graph View
- Visual map of every entity extracted from a document
- Nodes are entities (people, organisations, concepts, requirements, dates)
- Edges are relationships between them
- Clickable — click a node to ask a question about it

### 5.5 Chat History
- Every conversation is saved
- Users can return to past queries without redoing the work
- Sessions are searchable by document or keyword

### 5.6 User Accounts
- Sign up with email or Google
- Each user's documents and conversations are private
- Built for individual users in v1, team accounts in v2

---

## 6. The Knowledge Graph

This is what separates Mr.Summarizer from a standard document chatbot. When a document is processed, the system extracts not just facts but the **structure** of how things relate.

### Entity Types (Nodes)
| Entity | Examples |
|---|---|
| Person | Named individuals referenced in the document |
| Organisation | Companies, institutions, regulatory bodies |
| Location | Countries, cities, jurisdictions |
| Concept | Defined terms, technical ideas, named methodologies |
| Requirement | Rules, obligations, conditions, constraints |
| Date / Period | Deadlines, effective dates, durations |
| Product / Service | Named goods, offerings, systems |
| Event | Meetings, milestones, incidents |

### Relationship Types (Edges)
| Relationship | Example |
|---|---|
| REFERENCES | Clause 4 REFERENCES Schedule B |
| REQUIRES | Party A REQUIRES written notice from Party B |
| DEFINES | Section 1.1 DEFINES "Confidential Information" |
| APPLIES_TO | Restriction X APPLIES_TO Product Y in Jurisdiction Z |
| SUPERSEDES | Amendment 3 SUPERSEDES original Clause 7 |
| CONTRADICTS | Finding A CONTRADICTS Finding B from Section 4 |
| INVOLVES | Event X INVOLVES Organisation Y |
| RESTRICTS | Regulation Z RESTRICTS Substance W above threshold |

### Example Graph Traversal
Question: *"What happens if the supplier misses the delivery deadline?"*

```
delivery deadline
  └── REFERENCES ──► Clause 8 (Delivery Obligations)
                          └── REQUIRES ──► written notice within 5 days
                          └── APPLIES_TO ──► penalty clause
                                                └── DEFINES ──► 2% per week, capped at 10%
                                                └── REFERENCES ──► Force Majeure (Clause 14)
```

The system walks these edges, assembles the connected answer, and tells the user exactly what applies — with page references.

---

## 7. ML Pipeline

What happens when a PDF is uploaded:

```
Step 1 — Parse
  Extract text page by page using PyMuPDF
  Preserve page numbers for citations

Step 2 — NER (Named Entity Recognition)
  spaCy (en_core_web_trf) identifies entities:
  people, organisations, locations, dates, concepts

Step 3 — Relation Extraction
  REBEL model extracts structured (subject, relation, object) triplets
  e.g. (Clause 4) --[references]--> (Schedule B)

Step 4 — Classify
  Document type identified:
  contract / research paper / regulation / manual / report / general

Step 5 — Chunk
  Text split at semantic boundaries (topic changes)
  not at fixed character counts

Step 6 — Embed
  Each chunk embedded using BGE-base-en-v1.5
  Produces 768-dimensional vectors for similarity search

Step 7 — Store
  chunks table        → content + embedding + page reference
  entities table      → all extracted nodes
  relationships table → all extracted edges
```

---

## 8. Retrieval Pipeline

What happens when a user sends a question:

```
Step 1 — Route
  Classify query as semantic, relational, or hybrid

Step 2 — Retrieve
  Semantic   → vector similarity search across chunks
  Relational → recursive graph traversal across entities and relationships
  Hybrid     → both, results merged with Reciprocal Rank Fusion

Step 3 — Re-rank
  Cross-encoder scores all results
  Keeps the most relevant for the model

Step 4 — Compress
  Strip irrelevant sentences from each result

Step 5 — Generate
  Gemini 2.0 Flash produces a grounded answer
  Strictly based on the retrieved context
  Cites page numbers where information was found
```

---

## 9. Tech Stack

| Component | Technology |
|---|---|
| Frontend | Next.js 14, Tailwind CSS, shadcn/ui |
| Backend | FastAPI (Python) |
| LLM | Gemini 2.0 Flash (Google AI, free tier) |
| Embeddings | BGE-base-en-v1.5 (runs locally on server) |
| NER | spaCy en_core_web_trf |
| Relation Extraction | REBEL (Babelscape/rebel-large) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Database | Supabase (hosted PostgreSQL) |
| Vector Search | pgvector (Supabase built-in) |
| Graph Storage | PostgreSQL recursive CTEs (no separate graph DB) |
| Auth | Supabase Auth |
| File Storage | Supabase Storage |
| Frontend Deploy | Vercel |
| Backend Deploy | Railway |
| Repository | github.com/AdityaChaudhuri/RAG_Chatbot |

---

## 10. Database Schema (High Level)

| Table | What It Stores |
|---|---|
| `documents` | Uploaded PDFs — filename, type, entity count, upload date |
| `chunks` | Text chunks with 768-dim embeddings for vector search |
| `entities` | Extracted nodes — name, type, properties |
| `relationships` | Directed edges — source entity, relationship type, target entity |
| `chat_sessions` | Conversation groups |
| `chat_history` | Individual messages, partitioned by month |

---

## 11. API Endpoints

### Documents
| Method | Endpoint | What It Does |
|---|---|---|
| POST | `/documents/upload` | Upload and process a PDF |
| GET | `/documents` | List user's documents |
| DELETE | `/documents/{id}` | Delete document and all related data |
| POST | `/documents/{id}/summarise` | Full document summary |
| GET | `/documents/{id}/graph` | Return entity graph data for visualisation |

### Chat
| Method | Endpoint | What It Does |
|---|---|---|
| POST | `/chat/sessions` | Start a new conversation |
| GET | `/chat/sessions` | List all conversations |
| GET | `/chat/sessions/{id}/messages` | Get conversation history |
| POST | `/chat/sessions/{id}/messages` | Send a message, stream the response |

---

## 12. Frontend Pages

| Page | Purpose |
|---|---|
| `/` | Landing page — what the product does, who it is for |
| `/login` | Sign in with email or Google |
| `/library` | All uploaded documents |
| `/upload` | Drag and drop PDF upload |
| `/chat/{sessionId}` | Main chat interface with streaming responses |
| `/documents/{id}` | Document detail — entities, relationships, graph view, past sessions |

---

## 13. Build Phases

**Phase 1 — Database**
All SQL: tables, indexes, triggers, stored procedures, graph schema, RLS policies

**Phase 2 — Ingestion Pipeline**
PDF parsing, NER, REBEL relation extraction, classification, chunking, embedding, storage

**Phase 3 — Retrieval Pipeline**
Query router, vector search, graph traversal, multi-query, re-ranking, compression

**Phase 4 — Generation**
Gemini 2.0 Flash integration, prompt templates per document type and query mode

**Phase 5 — API**
All FastAPI routes, file upload, streaming

**Phase 6 — Frontend**
Next.js, auth, chat UI, document library, entity graph visualisation

**Phase 7 — Evaluation and Deployment**
RAGAS evaluation, performance tuning, Vercel + Railway deployment

---

## 14. Why This Works

- **Universal problem:** Everyone reads documents they don't fully have time to understand
- **Clear gap:** Existing tools (ChatGPT, Claude) read text linearly — they miss the structure
- **No domain expertise required:** Works on any PDF — legal, technical, academic, regulatory
- **Technical moat:** The knowledge graph accumulates structure that generic chatbots cannot replicate
- **Easy to demo:** Anyone can upload a PDF they already have and immediately see the value

---

*PRD version 3.0 — May 2026*
