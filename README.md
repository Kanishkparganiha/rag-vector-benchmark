# RAG Vector Benchmark

A comprehensive stress testing tool for **Pinecone vector database** with multi-modal RAG (Retrieval-Augmented Generation) support.

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [How Vector Databases Work](#how-vector-databases-work)
3. [Our Data: What We're Working With](#our-data-what-were-working-with)
4. [The Ingestion Pipeline](#the-ingestion-pipeline)
5. [Retrieval Methods](#retrieval-methods)
6. [Quick Start](#quick-start)
7. [Interactive Explorer (Notebook)](#interactive-explorer-notebook)
8. [Stress Tests](#stress-tests)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that enhances LLM responses by first retrieving relevant information from a knowledge base, then using that context to generate accurate answers.

### Why RAG Matters

```
Traditional LLM:
  User Question → LLM → Answer (based only on training data, may hallucinate)

RAG-Enhanced LLM:
  User Question → Search Knowledge Base → Retrieve Relevant Docs → LLM + Context → Accurate Answer
```

### RAG Benefits

| Problem | How RAG Solves It |
|---------|-------------------|
| LLM hallucinations | Grounds responses in actual documents |
| Outdated knowledge | Can query up-to-date information |
| Domain-specific needs | Uses your private/proprietary data |
| Cost | Smaller models + good retrieval = better results |

---

## How Vector Databases Work

### The Core Concept

Vector databases like **Pinecone** store data as high-dimensional vectors (arrays of numbers) and find similar items using mathematical distance calculations.

```
Traditional Database:
  "Find documents WHERE category = 'kubernetes'"
  → Exact match only

Vector Database:
  "Find documents SIMILAR TO 'how do I deploy containers?'"
  → Semantic similarity (understands meaning)
```

### How Pinecone Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        PINECONE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. INDEX CREATION                                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Index: "rag-benchmark"                                  │   │
│   │  Dimension: 384 (must match embedding size)              │   │
│   │  Metric: cosine (similarity measurement)                 │   │
│   │  Cloud: AWS us-east-1 (serverless, free tier)           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   2. VECTOR STORAGE                                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Each vector has:                                        │   │
│   │  • ID: "text_0001"                                       │   │
│   │  • Values: [0.023, -0.156, 0.789, ...] (384 floats)     │   │
│   │  • Metadata: {type: "text", topic: "kubernetes", ...}   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   3. SIMILARITY SEARCH                                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Query vector → Find K nearest neighbors                │   │
│   │  Uses Approximate Nearest Neighbor (ANN) algorithms     │   │
│   │  Returns: matches with similarity scores (0.0 to 1.0)   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Similarity Metrics

| Metric | Best For | How It Works |
|--------|----------|--------------|
| **Cosine** (we use this) | Text/NLP | Measures angle between vectors (ignores magnitude) |
| Euclidean | General | Straight-line distance between points |
| Dot Product | Recommendations | Measures alignment and magnitude |

---

## Our Data: What We're Working With

### Public Datasets (Default)

By default, we use **real public datasets** from HuggingFace for authentic benchmarking:

| Category | Dataset | Description |
|----------|---------|-------------|
| **Text** | `wikipedia` | Real Wikipedia articles (Simple English) |
| **Text** | `news` | AG News (World, Sports, Business, Sci/Tech) |
| **Text** | `financial` | Financial news with sentiment labels |
| **Chunks** | `squad` | Stanford QA Dataset (Wikipedia paragraphs) |
| **Chunks** | `arxiv` | ArXiv ML/AI paper abstracts |
| **Chunks** | `msmarco` | Microsoft search passages |
| **Chunks** | `stackoverflow` | Programming Q&A posts |
| **Chunks** | `pubmed` | Medical/scientific abstracts |
| **Chunks** | `github_readme` | Real GitHub repository READMEs |
| **Chunks** | `paul_graham` | Paul Graham's startup essays |
| **Images** | `coco` | Real photos with captions (COCO dataset) |
| **Images** | `diagrams` | Tech logos from Wikimedia Commons |

### Dataset Selection

```bash
# List all available datasets
python main.py --list-datasets

# Default: Wikipedia + SQuAD
python main.py

# Custom combination
python main.py --datasets wikipedia,arxiv,stackoverflow

# ML/AI focused
python main.py --datasets arxiv,pubmed,paul_graham

# Business/Finance
python main.py --datasets financial,news,wikipedia

# With real COCO images
python main.py --datasets wikipedia,squad,coco

# Use synthetic data instead (faster, no download)
python main.py --synthetic
```

### Data Types Overview

We load **3 types of data** from public sources (or generate synthetic):

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAW DATA TYPES                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. TEXT DOCUMENTS (1000 items)                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Short snippets (~50-100 words each)                     │    │
│  │  Example:                                                 │    │
│  │  {                                                        │    │
│  │    "id": "text_0042",                                     │    │
│  │    "content": "The Kubernetes component is essential     │    │
│  │               for container orchestration. It enables    │    │
│  │               teams to deploy applications while         │    │
│  │               maintaining system reliability...",        │    │
│  │    "topic": "Kubernetes",                                 │    │
│  │    "metadata": {                                          │    │
│  │      "author": "John Smith",                              │    │
│  │      "category": "tutorial",                              │    │
│  │      "difficulty": "intermediate"                         │    │
│  │    }                                                      │    │
│  │  }                                                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  2. DOCUMENT CHUNKS (200 items)                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Longer multi-paragraph content (~200-400 words)         │    │
│  │  Simulates chunked documentation (like splitting a       │    │
│  │  long PDF into retrievable sections)                     │    │
│  │                                                           │    │
│  │  Metadata includes:                                       │    │
│  │  • source_doc: Original document name                     │    │
│  │  • chunk_index: Position in original doc                  │    │
│  │  • section: "introduction", "implementation", etc.        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  3. IMAGES (50 items)                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Generated diagram images (PNG files)                     │    │
│  │  Types: architecture, flowchart, sequence, network, etc. │    │
│  │                                                           │    │
│  │  Stored in: images/                                       │    │
│  │  Files: architecture_001.png, flowchart_002.png, etc.    │    │
│  │                                                           │    │
│  │  Metadata includes:                                       │    │
│  │  • diagram_type: Type of diagram                          │    │
│  │  • description: Text description of the image             │    │
│  │  • filepath: Local path to PNG file                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Sample Real Data

**Wikipedia Article Example:**
```json
{
  "id": "wiki_0042",
  "source": "wikipedia",
  "title": "Kubernetes",
  "content": "Kubernetes is an open-source container orchestration
             system for automating software deployment, scaling,
             and management. Originally designed by Google...",
  "metadata": {
    "source": "wikipedia",
    "url": "https://simple.wikipedia.org/wiki/Kubernetes"
  }
}
```

**ArXiv Abstract Example:**
```json
{
  "id": "arxiv_0015",
  "source": "arxiv",
  "content": "We present a novel approach to neural machine translation
             using attention mechanisms that significantly improve
             translation quality on long sentences...",
  "metadata": {
    "source": "arxiv",
    "category": "cs.CL"
  }
}
```

**Stack Overflow Example:**
```json
{
  "id": "so_0089",
  "source": "stackoverflow",
  "title": "How to handle async/await errors in JavaScript?",
  "content": "Question: How to handle async/await errors in JavaScript?\n\n
             I'm trying to use try/catch with async functions but...",
  "metadata": {
    "source": "stackoverflow",
    "tags": "javascript,async-await,error-handling"
  }
}
```

### Topics Covered

Depending on datasets selected, content covers:
- **Wikipedia**: General knowledge, science, technology, history
- **ArXiv**: Machine learning, AI, NLP, computer vision
- **StackOverflow**: Programming, debugging, best practices
- **PubMed**: Medical research, biology, pharmaceuticals
- **Financial**: Stock market, earnings, business news
- **News**: World events, sports, business, technology

---

## The Ingestion Pipeline

### Overview: From Raw Data to Searchable Vectors

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: DATA GENERATION                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  data_generator.py                                        │    │
│  │                                                           │    │
│  │  • Generate 1000 text documents (tech snippets)           │    │
│  │  • Generate 200 document chunks (multi-paragraph)         │    │
│  │  • Generate 50 diagram images (PNG files)                 │    │
│  │                                                           │    │
│  │  Output: Python dictionaries with content + metadata      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│  STEP 2: EMBEDDING GENERATION                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  embeddings.py                                            │    │
│  │                                                           │    │
│  │  Text/Docs → SentenceTransformer → 384-dim vector        │    │
│  │  Images → CLIP Model → 384-dim vector                     │    │
│  │                                                           │    │
│  │  "How to configure Kubernetes"                            │    │
│  │       ↓                                                   │    │
│  │  [0.023, -0.156, 0.789, 0.445, ..., -0.234]              │    │
│  │   (384 floating point numbers)                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│  STEP 3: PINECONE UPSERT                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  pinecone_client.py                                       │    │
│  │                                                           │    │
│  │  Batch vectors into groups of 100                         │    │
│  │  Upload to Pinecone index                                 │    │
│  │                                                           │    │
│  │  Each upsert request:                                     │    │
│  │  {                                                        │    │
│  │    "id": "text_0001",                                     │    │
│  │    "values": [0.023, -0.156, ...],  // 384 floats        │    │
│  │    "metadata": {                                          │    │
│  │      "type": "text",                                      │    │
│  │      "topic": "Kubernetes",                               │    │
│  │      "content": "The Kubernetes component..."             │    │
│  │    }                                                      │    │
│  │  }                                                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│  STEP 4: INDEX READY                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Pinecone Cloud                                           │    │
│  │                                                           │    │
│  │  Total vectors: 1250                                      │    │
│  │  • 1000 text documents                                    │    │
│  │  • 200 document chunks                                    │    │
│  │  • 50 image embeddings                                    │    │
│  │                                                           │    │
│  │  Ready for similarity search!                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Embedding Models Used

| Data Type | Model | Why This Model |
|-----------|-------|----------------|
| **Text** | `all-MiniLM-L6-v2` | Fast, good quality, 384 dims (smaller = faster) |
| **Documents** | `all-MiniLM-L6-v2` | Same model for consistency |
| **Images** | `CLIP ViT-B-32` | Can embed images AND text in same space |

### How Embeddings Work

```python
# Text Embedding Example
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

text = "How to deploy Kubernetes pods"
embedding = model.encode(text)  # Returns 384-dim numpy array

# Similar texts have similar embeddings (close in vector space)
# "Kubernetes deployment guide" → similar vector
# "Best pizza recipes" → very different vector
```

### Batching for Performance

```
Why batch? Pinecone recommends batches of 100 vectors for optimal performance.

Single upsert:    1000 vectors × 1 request each = 1000 API calls (slow!)
Batched upsert:   1000 vectors ÷ 100 per batch = 10 API calls (fast!)
```

---

## Retrieval Methods

### Method 1: Basic Similarity Search

The simplest retrieval - find K most similar vectors to query.

```python
# User asks a question
query = "How do I configure Kubernetes for production?"

# Convert query to vector (same model as ingestion)
query_vector = model.encode(query)  # 384-dim vector

# Search Pinecone
results = index.query(
    vector=query_vector.tolist(),
    top_k=5,  # Return 5 most similar
    include_metadata=True
)

# Results ranked by similarity score (1.0 = identical, 0.0 = unrelated)
# Match 1: score=0.89, "Kubernetes production deployment guide..."
# Match 2: score=0.84, "Configuring K8s clusters for scale..."
# Match 3: score=0.79, "Container orchestration best practices..."
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    BASIC SIMILARITY SEARCH                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Query: "How do I configure Kubernetes?"                        │
│                    ↓                                             │
│   Embed: [0.12, -0.34, 0.56, ...]                               │
│                    ↓                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                 VECTOR SPACE                             │   │
│   │                                                          │   │
│   │         ★ Query                                          │   │
│   │        /|\                                               │   │
│   │       / | \                                              │   │
│   │      /  |  \                                             │   │
│   │     ●   ●   ●  ← Top 3 nearest neighbors                │   │
│   │                                                          │   │
│   │     ○  ○  ○  ○  ○  ← Other vectors (further away)       │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                    ↓                                             │
│   Return: Top K matches with scores                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Method 2: Filtered Search (Metadata Filtering)

Add filters to narrow down results by metadata.

```python
# Search only in text documents (not images or chunks)
results = index.query(
    vector=query_vector.tolist(),
    top_k=5,
    filter={
        "type": {"$eq": "text"}
    }
)

# Search only in advanced tutorials about specific topics
results = index.query(
    vector=query_vector.tolist(),
    top_k=5,
    filter={
        "type": {"$eq": "text"},
        "difficulty": {"$eq": "advanced"},
        "topic": {"$in": ["Kubernetes", "Docker", "Microservices"]}
    }
)
```

**Available Filter Operators:**
| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equals | `{"type": {"$eq": "text"}}` |
| `$ne` | Not equals | `{"type": {"$ne": "image"}}` |
| `$in` | In list | `{"topic": {"$in": ["K8s", "Docker"]}}` |
| `$gt`, `$gte` | Greater than | `{"word_count": {"$gt": 100}}` |
| `$lt`, `$lte` | Less than | `{"word_count": {"$lt": 500}}` |

### Method 3: Multi-Modal Search (Images + Text)

Search images using text queries via CLIP embeddings.

```python
# CLIP can embed both images and text in the SAME vector space
# This means text can find similar images!

query = "Show me architecture diagrams"

# Use CLIP's text encoder (not sentence-transformers)
clip_text_vector = clip_model.encode_text(query)

# Search in image vectors
results = index.query(
    vector=clip_text_vector.tolist(),
    top_k=5,
    filter={"type": {"$eq": "image"}}
)

# Returns: architecture_001.png, architecture_005.png, etc.
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIP MULTI-MODAL SEARCH                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   CLIP creates a SHARED vector space for images and text        │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Text: "a cat sitting"  ───→  [0.2, 0.5, ...]          │   │
│   │                                      ↓                   │   │
│   │                               Similar vectors!           │   │
│   │                                      ↑                   │   │
│   │   Image: 🐱 (photo of cat)  ───→  [0.21, 0.48, ...]     │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Use case: "Find architecture diagrams" → returns PNG files    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Method 4: Hybrid Search (Cross-Type)

Search across all content types simultaneously.

```python
def hybrid_search(query: str, top_k: int = 10):
    query_vector = model.encode(query)

    # Get all matches without type filter
    all_matches = index.query(
        vector=query_vector.tolist(),
        top_k=top_k * 3  # Get more, then organize
    )

    # Organize by type
    results = {
        "text": [],
        "document_chunk": [],
        "image": []
    }

    for match in all_matches:
        content_type = match.metadata.get("type")
        if len(results[content_type]) < top_k:
            results[content_type].append(match)

    return results

# Returns organized results from all content types
# {
#   "text": [top 10 text matches],
#   "document_chunk": [top 10 chunk matches],
#   "image": [top 10 image matches]
# }
```

### Method 5: RAG with Generation

Complete pipeline: Retrieve context, then generate answer.

```python
def rag_query(query: str, top_k: int = 5):
    # Step 1: Retrieve relevant documents
    query_vector = model.encode(query)
    matches = index.query(vector=query_vector.tolist(), top_k=top_k)

    # Step 2: Build context from retrieved docs
    context = "\n\n".join([
        match.metadata["content"]
        for match in matches
    ])

    # Step 3: Generate answer using LLM + context
    prompt = f"""Based on the following documents, answer the question.

Documents:
{context}

Question: {query}

Answer:"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User: "How do I set up Kubernetes autoscaling?"               │
│                         ↓                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  RETRIEVAL (Pinecone)                                    │   │
│   │  Query → Embed → Search → Top 5 relevant docs            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                         ↓                                        │
│   Retrieved Context:                                             │
│   • "Kubernetes HPA (Horizontal Pod Autoscaler) enables..."     │
│   • "Configure autoscaling with kubectl autoscale..."           │
│   • "Best practices for K8s scaling include..."                 │
│                         ↓                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  GENERATION (OpenAI/LLM)                                 │   │
│   │  Prompt = Context + Question                             │   │
│   │  → LLM generates informed answer                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                         ↓                                        │
│   Answer: "To set up Kubernetes autoscaling, you'll need to:    │
│           1. Enable the metrics server...                        │
│           2. Create an HPA resource...                           │
│           3. Configure min/max replicas..."                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Retrieval Methods Comparison

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Basic Similarity** | General search | Simple, fast | May return irrelevant types |
| **Filtered Search** | Specific content type | Precise, controlled | Requires known metadata |
| **Multi-Modal (CLIP)** | Image search with text | Cross-modal, intuitive | Separate model needed |
| **Hybrid Search** | Comprehensive results | Best coverage | More complex, slower |
| **RAG + Generation** | Q&A systems | Natural answers | Requires LLM API |

---

## Quick Start

### 1. Install Dependencies

```bash
cd rag-vector-benchmark
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your Pinecone API key
```

Get your free Pinecone API key at: https://www.pinecone.io/

### 3. Run the Benchmark

```bash
# Full benchmark (data generation + stress tests + visualizations)
python main.py

# Quick demo with minimal data
python main.py --demo

# Ingest data only — uses HuggingFace datasets (requires internet)
python main.py --ingest-only

# Ingest with synthetic data — fast, no HuggingFace token needed
python main.py --ingest-only --synthetic

# Just run stress tests (assumes data already ingested)
python main.py --test-only
```

> **Tip:** Use `--synthetic` for the first run or when HuggingFace rate limits apply.
> Synthetic data covers the same topics (Kubernetes, API, Microservices, etc.) without
> requiring any external downloads.

---

## Companion Project

This repo handles **ingestion only** — embedding and storing vectors in Pinecone.

For intelligent query routing, self-correcting retrieval, and LLM-powered answer
generation on top of this index, see the companion project:

**[langgraph-rag-agent](../langgraph-rag-agent/)** — LangGraph agentic RAG with
`llama3.2:3b` via Ollama. Reads from the same `rag-benchmark` Pinecone index.

---

## Interactive Explorer (Notebook)

Use the Jupyter notebook for **interactive querying and exploration** of your Pinecone index.

### Launch the Notebook

```bash
cd rag-vector-benchmark
jupyter notebook explore_pinecone.ipynb
```

### What You Can Do

| Feature | Description |
|---------|-------------|
| **View Index Stats** | See total vectors, dimensions, index fullness |
| **Browse Data** | Sample what's stored in your index |
| **Natural Language Search** | Query with `search("your question here")` |
| **Filter by Type** | Search only text, chunks, or images |
| **RAG Pipeline Demo** | Test retrieval with/without LLM generation |
| **Similarity Analysis** | Compare query embeddings visually |

### Quick Examples

```python
# Basic search
search("machine learning neural networks")

# Filter by data type
search("deep learning", filter_type="document_chunk")

# Search only images (if COCO loaded)
search("person with dog", filter_type="image")

# RAG retrieval (no LLM)
rag_retrieve("explain transformer architecture")

# Full RAG with answer (needs OpenAI key)
rag_answer("What is attention mechanism?")

# Check index stats
client.get_stats()
```

### How Pinecone Works (No Docker!)

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR LOCAL MACHINE                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Your Python │───▶│  Pinecone   │───▶│   HTTPS     │─────────┼──▶ Pinecone Cloud
│  │   Script    │    │   Client    │    │  Requests   │         │    (AWS/GCP servers)
│  └─────────────┘    └─────────────┘    └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘

Pinecone is a MANAGED CLOUD SERVICE:
- No Docker container needed
- No local installation
- Data stored on Pinecone's servers
- Access via API key over internet
- Free tier: 100K vectors, 1 index
```

### Embeddings Are NOT Reversible

```
Q: How do we get text back from embeddings?
A: We don't decode embeddings!

STORAGE (what Pinecone holds):
┌─────────────────────────────────────────────────────────────┐
│  id: "wiki_0042"                                            │
│  values: [0.023, -0.156, 0.089, ... 384 floats]  ← EMBEDDING│
│  metadata: {                                                │
│    "type": "text",                                          │
│    "topic": "wikipedia",                                    │
│    "content": "Kubernetes is an open-source..." ← ORIGINAL │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘

- Embeddings = search index (for finding similar items)
- Metadata.content = actual text (returned with results)
- You CANNOT decode [0.023, -0.156, ...] back to text
```

---

## Project Structure

```
rag-vector-benchmark/
├── main.py               # Main entry point (CLI)
├── explore_pinecone.ipynb # Interactive Jupyter notebook
├── config.py             # Configuration settings
├── data_loader.py        # Load public datasets (HuggingFace)
├── data_generator.py     # Generate synthetic data (fallback)
├── embeddings.py         # Embedding models (text, docs, images)
├── pinecone_client.py    # Pinecone operations
├── rag_pipeline.py       # RAG retrieval and generation
├── stress_test.py        # Stress testing scenarios
├── visualizations.py     # Charts and dashboards
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── ARCHITECTURE.md      # System design documentation
├── data/                # Downloaded/cached data
├── images/              # Downloaded/generated images
└── outputs/             # Results and visualizations
```

---

## Stress Tests

### 1. Concurrent Queries
Tests query performance with 10, 25, and 50 simultaneous queries.

### 2. Batch Insert Performance
Measures insert throughput with batch sizes of 10, 50, and 100 vectors.

### 3. Query Latency Under Load
Sustained load test at 5 QPS for 30 seconds.

### 4. Top-K Comparison
Compares retrieval performance for k=5, k=10, k=20.

---

## Sample Output

After running the benchmark, you'll find:

```
outputs/
├── dashboard.html          # Interactive Plotly dashboard
├── latency_comparison.png  # Latency bar chart
├── throughput_chart.png    # Throughput comparison
├── latency_distribution.png # Histogram of latencies
├── concurrent_scaling.png  # Scaling analysis
├── k_value_comparison.png  # Top-K comparison
├── summary_report.txt      # Text summary
└── results.json            # Raw JSON data
```

---

## Configuration

Edit `config.py` to customize:

```python
# Dataset sizes
NUM_TEXT_DOCUMENTS = 1000
NUM_DOCUMENT_CHUNKS = 200
NUM_IMAGES = 50

# Stress test parameters
CONCURRENT_QUERIES = [10, 25, 50]
BATCH_SIZES = [10, 50, 100]
TOP_K_VALUES = [5, 10, 20]
```

---

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Avg Latency** | Mean query response time |
| **P50/P95/P99** | Percentile latencies (P95 = 95% of queries faster than this) |
| **QPS** | Queries per second (throughput) |
| **Error Rate** | Percentage of failed requests |

---

## Requirements

- Python 3.8+
- Pinecone free tier account
- ~2GB RAM for embedding models
- ~500MB disk space for data and outputs

---

## Optional: OpenAI Integration

For AI-generated responses in the RAG demo:

```bash
# Add to .env
OPENAI_API_KEY=your-openai-key
```

Without OpenAI, the demo will show mock responses with retrieved context.

---

## Tips for Presentations

1. **Open `outputs/dashboard.html`** - Interactive charts you can zoom/hover
2. **Use `outputs/summary_report.txt`** - Quick stats overview
3. **PNG charts** - Copy to slides directly
4. **Explain the pipeline** - Use the diagrams in this README

---

## Troubleshooting

### "PINECONE_API_KEY not set"
```bash
cp .env.example .env
# Edit .env with your API key
```

### "ImportError: cannot import name 'Pinecone'"
The package was renamed from `pinecone-client` to `pinecone`. Fix:
```bash
pip uninstall pinecone-client -y
pip install pinecone>=5.0.0
```

### "Index not found"
```bash
python main.py --recreate-index
```

### Slow embedding generation
- First run downloads models (~500MB)
- Subsequent runs use cached models
- Use `--demo` for faster iteration

### Out of memory
Reduce dataset sizes in `config.py`:
```python
NUM_TEXT_DOCUMENTS = 500
NUM_DOCUMENT_CHUNKS = 100
NUM_IMAGES = 25
```

---

## License

MIT License - Free to use and modify.
