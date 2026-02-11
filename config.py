"""Configuration for RAG Vector Benchmark."""

import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-api-key-here")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-benchmark")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # Free tier region

# Embedding Models
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 384 dims
DOCUMENT_EMBEDDING_MODEL = "all-mpnet-base-v2"  # Better quality, 768 dims
IMAGE_EMBEDDING_MODEL = "ViT-B-32"  # CLIP model
IMAGE_EMBEDDING_PRETRAINED = "openai"

# Dimension for Pinecone (using smaller model for free tier)
EMBEDDING_DIMENSION = 384  # Match TEXT_EMBEDDING_MODEL

# Dataset Sizes (laptop-friendly)
NUM_TEXT_DOCUMENTS = 1000
NUM_DOCUMENT_CHUNKS = 200
NUM_IMAGES = 50
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Stress Test Configuration
CONCURRENT_QUERIES = [10, 25, 50]
BATCH_SIZES = [10, 50, 100]
TOP_K_VALUES = [5, 10, 20]
NUM_STRESS_ITERATIONS = 5

# Output paths
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
IMAGES_DIR = "images"
