# RAG Vector Benchmark - Architecture & Design

This document provides a visual overview of the system architecture, component design, and execution flows.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Component Overview](#component-overview)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Execution Flows](#execution-flows)
5. [Embedding Pipeline Details](#embedding-pipeline-details)
6. [Stress Testing Architecture](#stress-testing-architecture)
7. [Technology Stack](#technology-stack)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAG VECTOR BENCHMARK SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                              USER INTERFACE                              │    │
│  │                                                                          │    │
│  │    $ python main.py [--demo] [--datasets wikipedia,arxiv] [--synthetic] │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                          │
│                                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                           ORCHESTRATION LAYER                            │    │
│  │                              (main.py)                                   │    │
│  │                                                                          │    │
│  │   • Parse CLI arguments                                                  │    │
│  │   • Coordinate pipeline phases                                           │    │
│  │   • Handle errors and reporting                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                          │
│           ┌───────────────────────────┼───────────────────────────┐              │
│           │                           │                           │              │
│           ▼                           ▼                           ▼              │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐        │
│  │  DATA LAYER     │       │  SEARCH LAYER   │       │  TEST LAYER     │        │
│  │                 │       │                 │       │                 │        │
│  │ data_generator  │       │ rag_pipeline    │       │ stress_test     │        │
│  │ embeddings      │       │ pinecone_client │       │ visualizations  │        │
│  └─────────────────┘       └─────────────────┘       └─────────────────┘        │
│           │                           │                           │              │
│           └───────────────────────────┼───────────────────────────┘              │
│                                       │                                          │
│                                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          EXTERNAL SERVICES                               │    │
│  │                                                                          │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │    │
│  │   │   PINECONE   │    │   OPENAI     │    │  LOCAL ML    │              │    │
│  │   │   (Vector    │    │   (LLM Gen)  │    │  (Embeddings)│              │    │
│  │   │   Database)  │    │   Optional   │    │              │              │    │
│  │   └──────────────┘    └──────────────┘    └──────────────┘              │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Overview

### Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            MODULE DEPENDENCIES                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                              ┌──────────────┐                                    │
│                              │   main.py    │                                    │
│                              │  (entry pt)  │                                    │
│                              └──────┬───────┘                                    │
│                                     │                                            │
│               ┌─────────────────────┼─────────────────────┐                      │
│               │                     │                     │                      │
│               ▼                     ▼                     ▼                      │
│      ┌────────────────┐   ┌────────────────┐   ┌────────────────┐               │
│      │  data_loader   │   │  rag_pipeline  │   │  stress_test   │               │
│      │ (public data)  │   │                │   │                │               │
│      │ data_generator │   │                │   │                │               │
│      │ (synthetic)    │   │                │   │                │               │
│      └───────┬────────┘   └───────┬────────┘   └───────┬────────┘               │
│              │                    │                     │                        │
│              │            ┌───────┴───────┐             │                        │
│              │            │               │             │                        │
│              ▼            ▼               ▼             ▼                        │
│      ┌────────────────────────┐   ┌────────────────────────┐                    │
│      │      embeddings        │   │    pinecone_client     │                    │
│      │                        │   │                        │                    │
│      │  • EmbeddingManager    │   │  • PineconeClient      │                    │
│      │  • Text embeddings     │   │  • Index operations    │                    │
│      │  • Image embeddings    │   │  • Query methods       │                    │
│      └───────────┬────────────┘   └───────────┬────────────┘                    │
│                  │                            │                                  │
│                  └──────────┬─────────────────┘                                  │
│                             │                                                    │
│                             ▼                                                    │
│                    ┌────────────────┐                                            │
│                    │    config.py   │                                            │
│                    │                │                                            │
│                    │  • API keys    │                                            │
│                    │  • Model names │                                            │
│                    │  • Parameters  │                                            │
│                    └────────────────┘                                            │
│                                                                                  │
│                                                                                  │
│      ┌────────────────┐                                                          │
│      │ visualizations │  (standalone - called after stress tests)               │
│      │                │                                                          │
│      │  • Charts      │                                                          │
│      │  • Dashboard   │                                                          │
│      │  • Reports     │                                                          │
│      └────────────────┘                                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          COMPONENT RESPONSIBILITIES                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  config.py                                                               │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │  • Load environment variables (.env)                                     │    │
│  │  • Define Pinecone settings (API key, index name, region)               │    │
│  │  • Specify embedding model names and dimensions                          │    │
│  │  • Set dataset sizes (1000 texts, 200 chunks, 50 images)                │    │
│  │  • Configure stress test parameters                                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  data_loader.py  (PRIMARY - Public Datasets)                             │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │  • Load real data from HuggingFace datasets                              │    │
│  │  • Wikipedia, ArXiv, SQuAD, MS MARCO, StackOverflow                     │    │
│  │  • PubMed, Financial news, GitHub READMEs, Paul Graham                  │    │
│  │  • COCO images with captions                                             │    │
│  │  • Configurable via --datasets CLI argument                              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  data_generator.py  (FALLBACK - Synthetic Data)                          │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │  • Generate synthetic tech documentation (Faker library)                 │    │
│  │  • Create text documents with metadata                                   │    │
│  │  • Create document chunks (multi-paragraph)                              │    │
│  │  • Generate diagram images (PIL/Pillow)                                  │    │
│  │  • Use with --synthetic flag (faster, no download)                       │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  embeddings.py                                                           │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │  • EmbeddingManager class (lazy model loading)                          │    │
│  │  • embed_texts() - SentenceTransformer for short text                   │    │
│  │  • embed_documents() - SentenceTransformer for chunks                   │    │
│  │  • embed_images() - CLIP for image embeddings                           │    │
│  │  • embed_query() - Single query embedding                                │    │
│  │  • prepare_vectors_for_pinecone() - Format for upsert                   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  pinecone_client.py                                                      │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │  • PineconeClient class                                                  │    │
│  │  • create_index() - Create serverless index                             │    │
│  │  • connect() - Connect to existing index                                │    │
│  │  • upsert_vectors() - Batch insert with metrics                         │    │
│  │  • query() - Basic similarity search                                     │    │
│  │  • query_by_type() - Filtered search                                    │    │
│  │  • hybrid_query() - Cross-type search                                   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  rag_pipeline.py                                                         │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │  • RAGPipeline class (main interface)                                   │    │
│  │  • ingest() - Full data ingestion pipeline                              │    │
│  │  • search_text() - Text similarity search                               │    │
│  │  • search_images() - CLIP-based image search                            │    │
│  │  • hybrid_search() - Multi-type search                                  │    │
│  │  • generate_response() - OpenAI LLM generation                          │    │
│  │  • rag_query() - Complete retrieve + generate                           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  stress_test.py                                                          │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │  • StressTestSuite class                                                 │    │
│  │  • test_concurrent_queries() - ThreadPoolExecutor concurrency           │    │
│  │  • test_batch_insert_performance() - Upsert benchmarks                  │    │
│  │  • test_query_latency_under_load() - Sustained QPS test                 │    │
│  │  • test_different_k_values() - Top-K comparison                         │    │
│  │  • StressTestResult dataclass - Metrics container                       │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  visualizations.py                                                       │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │  • MetricsVisualizer class                                               │    │
│  │  • plot_latency_comparison() - Bar chart (matplotlib)                   │    │
│  │  • plot_throughput_chart() - Horizontal bar chart                       │    │
│  │  • plot_latency_distribution() - Histograms                             │    │
│  │  • plot_concurrent_scaling() - Line charts                              │    │
│  │  • create_interactive_dashboard() - Plotly HTML                         │    │
│  │  • generate_summary_report() - Text report                              │    │
│  │  • save_results_json() - Raw data export                                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagrams

### Complete Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE DATA PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   PHASE 1: DATA LOADING (from HuggingFace or Synthetic)                         │
│   ═════════════════════════════════════════════════════                          │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         HUGGINGFACE DATASETS                             │   │
│   │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  │   │
│   │  │ Wikipedia │ │   SQuAD   │ │   ArXiv   │ │  PubMed   │ │   COCO    │  │   │
│   │  │  articles │ │ passages  │ │ abstracts │ │ abstracts │ │  images   │  │   │
│   │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘  │   │
│   │        │             │             │             │             │         │   │
│   │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐               │   │
│   │  │  AG News  │ │StackOverf │ │ Financial │ │  GitHub   │               │   │
│   │  │  articles │ │    Q&A    │ │   news    │ │  READMEs  │               │   │
│   │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘               │   │
│   └────────┼─────────────┼─────────────┼─────────────┼─────────────────────┘   │
│            │             │             │             │                          │
│            └─────────────┴──────┬──────┴─────────────┘                          │
│                                 ▼                                               │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                       │
│   │    1000     │     │     200     │     │     50      │                       │
│   │    Text     │     │   Document  │     │   Images    │                       │
│   │  Documents  │     │   Chunks    │     │  (COCO/     │                       │
│   │ (wiki/news) │     │(squad/arxiv)│     │  Wikimedia) │                       │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                       │
│          │                   │                   │                               │
│          └───────────────────┼───────────────────┘                               │
│                              │                                                   │
│                              ▼                                                   │
│                    ┌─────────────────┐                                           │
│                    │   Raw Data      │                                           │
│                    │   Dictionary    │                                           │
│                    │                 │                                           │
│                    │ {               │                                           │
│                    │   "text_docs",  │                                           │
│                    │   "chunks",     │                                           │
│                    │   "images"      │                                           │
│                    │ }               │                                           │
│                    └────────┬────────┘                                           │
│                             │                                                    │
│   ══════════════════════════╪════════════════════════════════════════════════   │
│                             │                                                    │
│   PHASE 2: EMBEDDING        │                                                    │
│   ══════════════════        │                                                    │
│                             ▼                                                    │
│          ┌──────────────────┴──────────────────┐                                │
│          │                                      │                                │
│          ▼                                      ▼                                │
│   ┌─────────────────┐                   ┌─────────────────┐                     │
│   │ SentenceTransf. │                   │   CLIP Model    │                     │
│   │ all-MiniLM-L6   │                   │   ViT-B-32      │                     │
│   └────────┬────────┘                   └────────┬────────┘                     │
│            │                                     │                               │
│            ▼                                     ▼                               │
│   ┌─────────────────┐                   ┌─────────────────┐                     │
│   │  Text & Chunk   │                   │     Image       │                     │
│   │   Embeddings    │                   │   Embeddings    │                     │
│   │   (384-dim)     │                   │   (384-dim)     │                     │
│   └────────┬────────┘                   └────────┬────────┘                     │
│            │                                     │                               │
│            └──────────────────┬──────────────────┘                               │
│                               │                                                  │
│                               ▼                                                  │
│                    ┌─────────────────┐                                           │
│                    │  Vectors List   │                                           │
│                    │                 │                                           │
│                    │  [{             │                                           │
│                    │    id, values,  │                                           │
│                    │    metadata     │                                           │
│                    │  }, ...]        │                                           │
│                    │                 │                                           │
│                    │  1250 vectors   │                                           │
│                    └────────┬────────┘                                           │
│                             │                                                    │
│   ══════════════════════════╪════════════════════════════════════════════════   │
│                             │                                                    │
│   PHASE 3: STORAGE          │                                                    │
│   ════════════════          │                                                    │
│                             ▼                                                    │
│                    ┌─────────────────┐                                           │
│                    │  Batch Upsert   │                                           │
│                    │  (100 per batch)│                                           │
│                    └────────┬────────┘                                           │
│                             │                                                    │
│                             ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │                         PINECONE CLOUD                                   │   │
│   │                                                                          │   │
│   │    ┌────────────────────────────────────────────────────────────────┐   │   │
│   │    │                     Index: "rag-benchmark"                      │   │   │
│   │    │                                                                 │   │   │
│   │    │    Dimension: 384    Metric: cosine    Region: us-east-1       │   │   │
│   │    │                                                                 │   │   │
│   │    │    ┌───────────────────────────────────────────────────────┐   │   │   │
│   │    │    │  Vectors:                                              │   │   │   │
│   │    │    │                                                        │   │   │   │
│   │    │    │  text_0001  [0.02, -0.15, 0.78, ...]  {type: "text"}  │   │   │   │
│   │    │    │  text_0002  [0.11, -0.23, 0.65, ...]  {type: "text"}  │   │   │   │
│   │    │    │  ...                                                   │   │   │   │
│   │    │    │  chunk_0001 [0.08, -0.31, 0.42, ...]  {type: "chunk"} │   │   │   │
│   │    │    │  ...                                                   │   │   │   │
│   │    │    │  img_0001   [0.19, -0.05, 0.88, ...]  {type: "image"} │   │   │   │
│   │    │    │                                                        │   │   │   │
│   │    │    └───────────────────────────────────────────────────────┘   │   │   │
│   │    │                                                                 │   │   │
│   │    └────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Query Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              QUERY FLOW                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   USER QUERY                                                                     │
│   ══════════                                                                     │
│                                                                                  │
│   "How do I configure Kubernetes autoscaling?"                                   │
│                              │                                                   │
│                              ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         QUERY EMBEDDING                                  │   │
│   │                                                                          │   │
│   │   SentenceTransformer.encode("How do I configure...")                   │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   [0.156, -0.234, 0.789, 0.123, ..., -0.456]  (384 dimensions)          │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      PINECONE SIMILARITY SEARCH                          │   │
│   │                                                                          │   │
│   │   index.query(                                                           │   │
│   │       vector=[0.156, -0.234, ...],                                      │   │
│   │       top_k=5,                                                           │   │
│   │       include_metadata=True                                              │   │
│   │   )                                                                      │   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │                    APPROXIMATE NEAREST NEIGHBOR                  │   │   │
│   │   │                                                                  │   │   │
│   │   │   Query Vector ──→ Compare with all 1250 vectors                │   │   │
│   │   │                    using cosine similarity                       │   │   │
│   │   │                              │                                   │   │   │
│   │   │                              ▼                                   │   │   │
│   │   │                    Return top 5 most similar                     │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         SEARCH RESULTS                                   │   │
│   │                                                                          │   │
│   │   Match 1: score=0.92                                                    │   │
│   │   ├─ id: "chunk_0089"                                                    │   │
│   │   ├─ type: "document_chunk"                                              │   │
│   │   └─ content: "Kubernetes HPA enables automatic scaling..."             │   │
│   │                                                                          │   │
│   │   Match 2: score=0.87                                                    │   │
│   │   ├─ id: "text_0234"                                                     │   │
│   │   ├─ type: "text"                                                        │   │
│   │   └─ content: "Configure autoscaling with kubectl..."                   │   │
│   │                                                                          │   │
│   │   Match 3: score=0.84                                                    │   │
│   │   ├─ id: "text_0567"                                                     │   │
│   │   ├─ type: "text"                                                        │   │
│   │   └─ content: "Best practices for K8s pod scaling..."                   │   │
│   │                                                                          │   │
│   │   ... (2 more matches)                                                   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    OPTIONAL: LLM GENERATION                              │   │
│   │                                                                          │   │
│   │   Context = Retrieved document contents                                  │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │  OpenAI GPT-3.5-turbo                                            │   │   │
│   │   │                                                                  │   │   │
│   │   │  System: You are a helpful technical assistant.                  │   │   │
│   │   │                                                                  │   │   │
│   │   │  User: Based on these documents, answer:                         │   │   │
│   │   │        [Context from retrieved docs]                             │   │   │
│   │   │        Question: How do I configure Kubernetes autoscaling?      │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   Generated Answer:                                                      │   │
│   │   "To configure Kubernetes autoscaling, follow these steps:             │   │
│   │    1. Enable the metrics server in your cluster                         │   │
│   │    2. Create a Horizontal Pod Autoscaler (HPA) resource                 │   │
│   │    3. Set min/max replica counts and target CPU utilization..."         │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Execution Flows

### Full Benchmark Execution

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     FULL BENCHMARK EXECUTION FLOW                                │
│                        $ python main.py                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   START                                                                          │
│     │                                                                            │
│     ▼                                                                            │
│   ┌─────────────────────┐                                                        │
│   │ Check Environment   │                                                        │
│   │ • .env file exists? │                                                        │
│   │ • API key valid?    │                                                        │
│   └──────────┬──────────┘                                                        │
│              │                                                                   │
│              ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  PHASE 1: DATA GENERATION & INGESTION                    ~5-10 minutes  │   │
│   │                                                                          │   │
│   │  1.1 Generate Data                                                       │   │
│   │      ├─ generate_text_documents(1000)                                   │   │
│   │      ├─ generate_document_chunks(200)                                   │   │
│   │      └─ generate_images(50)                                             │   │
│   │                                                                          │   │
│   │  1.2 Create Embeddings                                                   │   │
│   │      ├─ Load SentenceTransformer model (first run: download ~90MB)      │   │
│   │      ├─ Load CLIP model (first run: download ~400MB)                    │   │
│   │      ├─ Embed 1000 texts → 1000 vectors                                 │   │
│   │      ├─ Embed 200 chunks → 200 vectors                                  │   │
│   │      └─ Embed 50 images → 50 vectors                                    │   │
│   │                                                                          │   │
│   │  1.3 Create/Connect Pinecone Index                                       │   │
│   │      ├─ Create index "rag-benchmark" (if not exists)                    │   │
│   │      └─ Wait for index ready                                            │   │
│   │                                                                          │   │
│   │  1.4 Upsert Vectors                                                      │   │
│   │      ├─ Batch size: 100 vectors per request                             │   │
│   │      ├─ 13 batches total                                                │   │
│   │      └─ Wait for indexing (10 seconds)                                  │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│              │                                                                   │
│              ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  PHASE 2: STRESS TESTING                                 ~3-5 minutes   │   │
│   │                                                                          │   │
│   │  2.1 Concurrent Queries Test                                             │   │
│   │      ├─ 10 concurrent queries × 5 iterations                            │   │
│   │      ├─ 25 concurrent queries × 5 iterations                            │   │
│   │      └─ 50 concurrent queries × 5 iterations                            │   │
│   │                                                                          │   │
│   │  2.2 Batch Insert Test                                                   │   │
│   │      ├─ Generate 200 test vectors                                       │   │
│   │      ├─ Insert with batch_size=10                                       │   │
│   │      ├─ Insert with batch_size=50                                       │   │
│   │      ├─ Insert with batch_size=100                                      │   │
│   │      └─ Cleanup test vectors                                            │   │
│   │                                                                          │   │
│   │  2.3 Top-K Comparison                                                    │   │
│   │      ├─ 50 queries with k=5                                             │   │
│   │      ├─ 50 queries with k=10                                            │   │
│   │      └─ 50 queries with k=20                                            │   │
│   │                                                                          │   │
│   │  2.4 Latency Under Load                                                  │   │
│   │      └─ 15 seconds @ 5 QPS sustained                                    │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│              │                                                                   │
│              ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  PHASE 3: RAG DEMONSTRATION                              ~1 minute      │   │
│   │                                                                          │   │
│   │  Run 4 sample queries:                                                   │   │
│   │  ├─ "How do I configure Kubernetes for production?"                     │   │
│   │  ├─ "What are the best practices for API authentication?"               │   │
│   │  ├─ "Explain database sharding strategies"                              │   │
│   │  └─ "Show me architecture diagrams for microservices"                   │   │
│   │                                                                          │   │
│   │  For each: Retrieve → Display results → Generate response               │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│              │                                                                   │
│              ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  PHASE 4: VISUALIZATION                                  ~30 seconds    │   │
│   │                                                                          │   │
│   │  Generate outputs:                                                       │   │
│   │  ├─ latency_comparison.png                                              │   │
│   │  ├─ throughput_chart.png                                                │   │
│   │  ├─ latency_distribution.png                                            │   │
│   │  ├─ concurrent_scaling.png                                              │   │
│   │  ├─ k_value_comparison.png                                              │   │
│   │  ├─ dashboard.html (interactive)                                        │   │
│   │  ├─ summary_report.txt                                                  │   │
│   │  └─ results.json                                                        │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│              │                                                                   │
│              ▼                                                                   │
│            END                                                                   │
│                                                                                  │
│   Total time: ~10-15 minutes (first run with model downloads)                   │
│               ~5-8 minutes (subsequent runs with cached models)                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Quick Demo Execution

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      QUICK DEMO EXECUTION FLOW                                   │
│                        $ python main.py --demo                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Reduced dataset for faster iteration:                                          │
│   • 50 text documents (instead of 1000)                                         │
│   • 20 document chunks (instead of 200)                                         │
│   • 10 images (instead of 50)                                                   │
│                                                                                  │
│   Reduced tests:                                                                 │
│   • 5 concurrent queries × 2 iterations                                         │
│   • k=[5, 10] only (no k=20)                                                    │
│   • No latency under load test                                                  │
│                                                                                  │
│   Total time: ~2-3 minutes                                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Embedding Pipeline Details

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING MODELS ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   TEXT EMBEDDING: all-MiniLM-L6-v2                                               │
│   ════════════════════════════════                                               │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   Input Text: "How to configure Kubernetes pods"                        │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │                      TOKENIZATION                                 │  │   │
│   │   │   ["How", "to", "configure", "Kubernetes", "pods"]               │  │   │
│   │   │                              │                                    │  │   │
│   │   │                              ▼                                    │  │   │
│   │   │   [101, 2129, 2000, 9530, 18169, 8962, 102]  (token IDs)         │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │                    TRANSFORMER LAYERS                             │  │   │
│   │   │                                                                   │  │   │
│   │   │   6 layers of self-attention + feed-forward                      │  │   │
│   │   │   Hidden size: 384                                                │  │   │
│   │   │   Attention heads: 12                                             │  │   │
│   │   │                                                                   │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │                      MEAN POOLING                                 │  │   │
│   │   │   Average all token embeddings → single 384-dim vector           │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   Output: [0.023, -0.156, 0.789, ..., -0.234]  (384 floats)             │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│                                                                                  │
│   IMAGE EMBEDDING: CLIP ViT-B-32                                                 │
│   ══════════════════════════════                                                 │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   Input: Image (e.g., architecture_001.png)                             │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │                    IMAGE PREPROCESSING                            │  │   │
│   │   │   Resize to 224×224, normalize pixel values                      │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │                   VISION TRANSFORMER (ViT)                        │  │   │
│   │   │                                                                   │  │   │
│   │   │   Split image into 32×32 patches (49 patches)                    │  │   │
│   │   │   Linear projection of patches                                    │  │   │
│   │   │   12 transformer layers                                           │  │   │
│   │   │   Output dimension: 512                                           │  │   │
│   │   │                                                                   │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │                   DIMENSION PROJECTION                            │  │   │
│   │   │   512 → 384 (truncate to match text embedding dimension)         │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   Output: [0.189, -0.045, 0.678, ..., -0.123]  (384 floats)             │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Stress Testing Architecture

### Concurrent Query Test Design

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      CONCURRENT QUERY TEST ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ThreadPoolExecutor(max_workers=N)                                              │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   Pre-generate N query vectors (to avoid embedding time in test)        │   │
│   │                                                                          │   │
│   │   query_vectors = [                                                      │   │
│   │       embed("How to implement API Gateway?"),                           │   │
│   │       embed("Kubernetes best practices"),                                │   │
│   │       embed("Database sharding strategies"),                             │   │
│   │       ...                                                                │   │
│   │   ]                                                                      │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   for iteration in range(5):  # 5 iterations per concurrency level      │   │
│   │       │                                                                  │   │
│   │       ▼                                                                  │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │              THREAD POOL (e.g., 10 workers)                      │   │   │
│   │   │                                                                  │   │   │
│   │   │   ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐              │   │   │
│   │   │   │Thread 1│ │Thread 2│ │Thread 3│ ... │Thread N│              │   │   │
│   │   │   │        │ │        │ │        │     │        │              │   │   │
│   │   │   │ query  │ │ query  │ │ query  │     │ query  │              │   │   │
│   │   │   │ vec[0] │ │ vec[1] │ │ vec[2] │     │ vec[N] │              │   │   │
│   │   │   │        │ │        │ │        │     │        │              │   │   │
│   │   │   └───┬────┘ └───┬────┘ └───┬────┘     └───┬────┘              │   │   │
│   │   │       │          │          │              │                    │   │   │
│   │   │       └──────────┴──────────┴──────────────┘                    │   │   │
│   │   │                          │                                       │   │   │
│   │   │                          ▼                                       │   │   │
│   │   │                    ┌──────────┐                                  │   │   │
│   │   │                    │ PINECONE │                                  │   │   │
│   │   │                    │   API    │                                  │   │   │
│   │   │                    └──────────┘                                  │   │   │
│   │   │                                                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                           │   │
│   │                              ▼                                           │   │
│   │   Collect results:                                                       │   │
│   │   • latencies[]  - time for each query                                  │   │
│   │   • success_count - queries that succeeded                              │   │
│   │   • error_count - queries that failed                                   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   Output Metrics:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  test_name: "concurrent_queries_10"                                      │   │
│   │  total_requests: 50 (10 concurrent × 5 iterations)                       │   │
│   │  successful_requests: 50                                                 │   │
│   │  avg_latency: 45.2ms                                                     │   │
│   │  p95_latency: 78.5ms                                                     │   │
│   │  throughput: 22.1 QPS                                                    │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Metrics Collection

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         METRICS COLLECTION STRUCTURE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   @dataclass                                                                     │
│   class StressTestResult:                                                        │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   test_name: str              # "concurrent_queries_10"                 │   │
│   │   total_requests: int         # 50                                      │   │
│   │   total_time: float           # 2.26 seconds                            │   │
│   │   successful_requests: int    # 50                                      │   │
│   │   failed_requests: int        # 0                                       │   │
│   │   latencies: List[float]      # [0.042, 0.038, 0.051, ...]             │   │
│   │   errors: List[str]           # []                                      │   │
│   │                                                                          │   │
│   │   ─────────────────────────────────────────────────────────────────     │   │
│   │   Computed Properties:                                                   │   │
│   │   ─────────────────────────────────────────────────────────────────     │   │
│   │                                                                          │   │
│   │   avg_latency    = mean(latencies)           # 0.0452                   │   │
│   │   p50_latency    = percentile(latencies, 50) # 0.0420                   │   │
│   │   p95_latency    = percentile(latencies, 95) # 0.0785                   │   │
│   │   p99_latency    = percentile(latencies, 99) # 0.0923                   │   │
│   │   throughput     = successful / total_time   # 22.1 QPS                 │   │
│   │   error_rate     = failed / total            # 0.0                      │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│                                                                                  │
│   Aggregated Results (after all tests):                                          │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   results = [                                                            │   │
│   │       StressTestResult("concurrent_queries_10", ...),                   │   │
│   │       StressTestResult("concurrent_queries_25", ...),                   │   │
│   │       StressTestResult("concurrent_queries_50", ...),                   │   │
│   │       StressTestResult("batch_insert_10", ...),                         │   │
│   │       StressTestResult("batch_insert_50", ...),                         │   │
│   │       StressTestResult("batch_insert_100", ...),                        │   │
│   │       StressTestResult("top_k_5", ...),                                 │   │
│   │       StressTestResult("top_k_10", ...),                                │   │
│   │       StressTestResult("top_k_20", ...),                                │   │
│   │       StressTestResult("latency_under_load_5qps", ...),                 │   │
│   │   ]                                                                      │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TECHNOLOGY STACK                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   LAYER              TECHNOLOGY           PURPOSE                                │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                  │
│   Vector Database    Pinecone             Cloud-hosted vector storage & search  │
│                      (Serverless)         Free tier: 1 index, 2GB storage       │
│                                                                                  │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                  │
│   Text Embeddings    SentenceTransformers Fast, high-quality text embeddings   │
│                      all-MiniLM-L6-v2     384 dimensions, ~90MB model           │
│                                                                                  │
│   Image Embeddings   OpenCLIP             Multi-modal embeddings                │
│                      ViT-B-32             Text and images in same space         │
│                                                                                  │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                  │
│   LLM Generation     OpenAI API           Response generation (optional)        │
│                      gpt-3.5-turbo        Context-aware answers                 │
│                                                                                  │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                  │
│   Data Sources       HuggingFace          Real public datasets (primary)        │
│                      datasets lib         Wikipedia, ArXiv, SQuAD, COCO, etc.   │
│                                                                                  │
│                      Faker                Synthetic text (fallback)             │
│                      Pillow               Image generation (fallback)           │
│                                                                                  │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                  │
│   Visualization      Matplotlib           Static PNG charts                     │
│                      Plotly               Interactive HTML dashboard            │
│                                                                                  │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                  │
│   Concurrency        ThreadPoolExecutor   Parallel query execution              │
│                      (Python stdlib)      Stress testing                        │
│                                                                                  │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                  │
│   Core               Python 3.8+          Runtime                               │
│                      NumPy                Numerical operations                  │
│                      PyTorch              ML framework (for models)             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DEPENDENCY RELATIONSHIPS                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                              ┌─────────────┐                                     │
│                              │   Python    │                                     │
│                              │   3.8+      │                                     │
│                              └──────┬──────┘                                     │
│                                     │                                            │
│           ┌─────────────────────────┼─────────────────────────┐                  │
│           │                         │                         │                  │
│           ▼                         ▼                         ▼                  │
│   ┌───────────────┐         ┌───────────────┐         ┌───────────────┐         │
│   │    PyTorch    │         │     NumPy     │         │    Pillow     │         │
│   │    (2.0+)     │         │    (1.24+)    │         │    (9.0+)     │         │
│   └───────┬───────┘         └───────┬───────┘         └───────────────┘         │
│           │                         │                                            │
│           ▼                         │                                            │
│   ┌───────────────┐                 │                                            │
│   │ Transformers  │                 │                                            │
│   │   (4.30+)     │                 │                                            │
│   └───────┬───────┘                 │                                            │
│           │                         │                                            │
│     ┌─────┴─────┐                   │                                            │
│     │           │                   │                                            │
│     ▼           ▼                   │                                            │
│ ┌─────────┐ ┌─────────┐             │                                            │
│ │Sentence │ │OpenCLIP │             │                                            │
│ │Transf.  │ │ (2.20+) │             │                                            │
│ │ (2.2+)  │ │         │             │                                            │
│ └─────────┘ └─────────┘             │                                            │
│                                     │                                            │
│                                     ▼                                            │
│                             ┌───────────────┐                                    │
│                             │  Matplotlib   │                                    │
│                             │    (3.7+)     │                                    │
│                             └───────────────┘                                    │
│                                                                                  │
│   Independent:                                                                   │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │  Pinecone   │  │   OpenAI    │  │   Plotly    │  │   Faker     │            │
│   │  (3.0+)     │  │   (1.0+)    │  │   (5.15+)   │  │  (18.0+)    │            │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File I/O Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FILE I/O SUMMARY                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUTS                                                                         │
│   ──────                                                                         │
│   .env                          Environment variables (API keys)                │
│   config.py                     Configuration parameters                         │
│   --datasets CLI arg            Dataset selection (wikipedia,arxiv,etc.)        │
│                                                                                  │
│   DOWNLOADED/CACHED (from HuggingFace)                                           │
│   ────────────────────────────────────                                           │
│   ~/.cache/huggingface/         Cached datasets (~100MB-1GB depending on        │
│                                 datasets selected)                               │
│   data/cache/                   Local data cache                                │
│                                                                                  │
│   GENERATED/DOWNLOADED (intermediate)                                            │
│   ───────────────────────────────────                                            │
│   images/*.jpg                  COCO images (if --datasets coco)                │
│   images/*.png                  Wikimedia tech logos/diagrams                   │
│                                                                                  │
│   OUTPUTS                                                                        │
│   ───────                                                                        │
│   outputs/                                                                       │
│   ├── latency_comparison.png    Bar chart: avg/p95/p99 latency by test         │
│   ├── throughput_chart.png      Horizontal bar: QPS by test                    │
│   ├── latency_distribution.png  Histograms: latency distribution (4 tests)     │
│   ├── concurrent_scaling.png    Line chart: latency/throughput vs concurrency  │
│   ├── k_value_comparison.png    Bar chart: latency by top-K value              │
│   ├── dashboard.html            Interactive Plotly dashboard                    │
│   ├── summary_report.txt        Human-readable text summary                     │
│   └── results.json              Raw JSON data for further analysis              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Key Numbers

| Metric | Value |
|--------|-------|
| Total vectors | 1,250 |
| Text documents | 1,000 |
| Document chunks | 200 |
| Images | 50 |
| Embedding dimension | 384 |
| Batch size (upsert) | 100 |
| Concurrent query levels | 10, 25, 50 |
| Top-K values tested | 5, 10, 20 |
| Stress test iterations | 5 |
| Model download size | ~500MB (first run) |
| Estimated runtime | 5-15 minutes |

---

## Available Public Datasets

| Dataset | Type | Source | Description |
|---------|------|--------|-------------|
| `wikipedia` | Text | HuggingFace | Real Wikipedia articles (Simple English) |
| `news` | Text | HuggingFace | AG News (World, Sports, Business, Sci/Tech) |
| `financial` | Text | HuggingFace | Financial news with sentiment labels |
| `squad` | Chunks | HuggingFace | Stanford QA Dataset paragraphs |
| `arxiv` | Chunks | HuggingFace | ArXiv ML/AI paper abstracts |
| `msmarco` | Chunks | HuggingFace | Microsoft search passages |
| `stackoverflow` | Chunks | HuggingFace | Programming Q&A posts |
| `pubmed` | Chunks | HuggingFace | Medical/scientific abstracts |
| `github_readme` | Chunks | HuggingFace | Real GitHub repository READMEs |
| `paul_graham` | Chunks | HuggingFace | Paul Graham's startup essays |
| `coco` | Images | HuggingFace | Real photos with captions |
| `diagrams` | Images | Wikimedia | Tech logos and diagrams |

### Dataset Selection Examples

```bash
# List all available datasets
python main.py --list-datasets

# Default (Wikipedia + SQuAD)
python main.py

# ML/AI focused
python main.py --datasets arxiv,pubmed,paul_graham

# Business/Finance
python main.py --datasets financial,news,wikipedia

# Full stack developer
python main.py --datasets stackoverflow,github_readme,wikipedia

# With real photos
python main.py --datasets wikipedia,squad,coco

# Synthetic only (faster, no download)
python main.py --synthetic
```
