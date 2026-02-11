# RAG Vector Benchmark - Simple Version

A **beginner-friendly** single notebook to learn RAG with Pinecone.

## What You'll Learn

1. Load data from Wikipedia
2. Convert text to embeddings (vectors)
3. Store vectors in Pinecone cloud database
4. Search with natural language
5. Run simple stress tests

## Quick Start

### 1. Get a Pinecone API Key (Free)

1. Go to https://www.pinecone.io/
2. Sign up (free tier = 100K vectors)
3. Copy your API key

### 2. Install Requirements

```bash
pip install pinecone-client sentence-transformers datasets tqdm matplotlib
```

### 3. Run the Notebook

```bash
jupyter notebook rag_benchmark_simple.ipynb
```

### 4. Add Your API Key

In the notebook, replace:
```python
PINECONE_API_KEY = "your-api-key-here"
```

## What's in the Notebook?

| Section | What it Does |
|---------|--------------|
| Step 1 | Configuration |
| Step 2 | Load Wikipedia data |
| Step 3 | Create embeddings |
| Step 4 | Connect to Pinecone |
| Step 5 | Upload vectors |
| Step 6 | Search with natural language |
| Step 7 | Run stress test |
| Step 8 | Visualize results |

## Key Concepts Explained

### What is an Embedding?

```
"I love dogs"  →  [0.23, -0.15, 0.89, ...] (384 numbers)
"I adore puppies"  →  [0.21, -0.14, 0.87, ...] (similar numbers!)
```

Text with similar meaning = similar numbers = we can find related content!

### What is Pinecone?

- Cloud database for vectors
- Search by meaning, not keywords
- No installation needed
- Free tier: 100K vectors

### What is RAG?

```
User asks: "How does X work?"
      ↓
Search your data for relevant docs
      ↓
Give docs + question to LLM
      ↓
Get accurate answer!
```

## Full Version

Want more features? Check out the `master` branch:
- Multiple dataset sources (ArXiv, PubMed, news, etc.)
- Image embeddings with CLIP
- Concurrent stress tests
- Interactive dashboard
- LLM generation

```bash
git checkout master
```

## Files

```
simple branch/
├── rag_benchmark_simple.ipynb  # ← The main notebook
├── README_SIMPLE.md            # ← This file
└── requirements_simple.txt     # ← Minimal dependencies
```
