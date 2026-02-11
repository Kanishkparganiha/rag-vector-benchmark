"""Embedding generation for text, documents, and images."""

import os
from typing import List, Dict, Union, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

import config


class EmbeddingManager:
    """Manages different embedding models for text, documents, and images."""

    def __init__(self, use_gpu: bool = False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self._text_model = None
        self._doc_model = None
        self._image_model = None
        self._image_preprocess = None
        self._tokenizer = None

    @property
    def text_model(self):
        """Lazy load text embedding model."""
        if self._text_model is None:
            print(f"Loading text model: {config.TEXT_EMBEDDING_MODEL}")
            from sentence_transformers import SentenceTransformer
            self._text_model = SentenceTransformer(config.TEXT_EMBEDDING_MODEL, device=self.device)
        return self._text_model

    @property
    def doc_model(self):
        """Lazy load document embedding model."""
        if self._doc_model is None:
            print(f"Loading document model: {config.DOCUMENT_EMBEDDING_MODEL}")
            from sentence_transformers import SentenceTransformer
            self._doc_model = SentenceTransformer(config.DOCUMENT_EMBEDDING_MODEL, device=self.device)
        return self._doc_model

    def _load_clip_model(self):
        """Load CLIP model for image embeddings."""
        if self._image_model is None:
            print(f"Loading CLIP model: {config.IMAGE_EMBEDDING_MODEL}")
            try:
                import open_clip
                self._image_model, _, self._image_preprocess = open_clip.create_model_and_transforms(
                    config.IMAGE_EMBEDDING_MODEL,
                    pretrained=config.IMAGE_EMBEDDING_PRETRAINED
                )
                self._tokenizer = open_clip.get_tokenizer(config.IMAGE_EMBEDDING_MODEL)
                self._image_model = self._image_model.to(self.device)
                self._image_model.eval()
            except ImportError:
                print("Warning: open_clip not available, using fallback for image embeddings")
                self._image_model = "fallback"

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for short text snippets."""
        if show_progress:
            print(f"Generating text embeddings for {len(texts)} texts...")

        embeddings = self.text_model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def embed_documents(self, documents: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for longer document chunks."""
        if show_progress:
            print(f"Generating document embeddings for {len(documents)} chunks...")

        # For documents, we use the same model but could use a different one
        # Using text model for consistency in dimension (384)
        embeddings = self.text_model.encode(
            documents,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def embed_images(self, image_paths: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for images using CLIP."""
        self._load_clip_model()

        if show_progress:
            print(f"Generating image embeddings for {len(image_paths)} images...")

        if self._image_model == "fallback":
            # Fallback: generate random embeddings with correct dimension
            print("Using fallback random embeddings for images")
            return np.random.randn(len(image_paths), config.EMBEDDING_DIMENSION).astype(np.float32)

        embeddings = []
        iterator = tqdm(image_paths) if show_progress else image_paths

        with torch.no_grad():
            for img_path in iterator:
                try:
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = self._image_preprocess(image).unsqueeze(0).to(self.device)
                    image_features = self._image_model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    # Reduce dimension to match text embeddings (384)
                    # Simple linear projection or truncation
                    emb = image_features.cpu().numpy().flatten()
                    if len(emb) > config.EMBEDDING_DIMENSION:
                        emb = emb[:config.EMBEDDING_DIMENSION]
                    elif len(emb) < config.EMBEDDING_DIMENSION:
                        emb = np.pad(emb, (0, config.EMBEDDING_DIMENSION - len(emb)))

                    embeddings.append(emb)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    embeddings.append(np.zeros(config.EMBEDDING_DIMENSION))

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        embedding = self.text_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]

    def embed_image_query(self, query: str) -> np.ndarray:
        """Generate CLIP text embedding for image search."""
        self._load_clip_model()

        if self._image_model == "fallback":
            return self.embed_query(query)

        with torch.no_grad():
            text_tokens = self._tokenizer([query]).to(self.device)
            text_features = self._image_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            emb = text_features.cpu().numpy().flatten()
            if len(emb) > config.EMBEDDING_DIMENSION:
                emb = emb[:config.EMBEDDING_DIMENSION]
            elif len(emb) < config.EMBEDDING_DIMENSION:
                emb = np.pad(emb, (0, config.EMBEDDING_DIMENSION - len(emb)))

            return emb.astype(np.float32)


def prepare_vectors_for_pinecone(
    data: Dict[str, List[Dict]],
    embedding_manager: EmbeddingManager
) -> List[Dict]:
    """Prepare all vectors for Pinecone ingestion."""
    vectors = []

    # Process text documents
    print("\n--- Processing Text Documents ---")
    texts = [doc["content"] for doc in data["text_documents"]]
    text_embeddings = embedding_manager.embed_texts(texts)

    for doc, emb in zip(data["text_documents"], text_embeddings):
        vectors.append({
            "id": doc["id"],
            "values": emb.tolist(),
            "metadata": {
                "type": "text",
                "topic": doc.get("topic", doc.get("source", "unknown")),
                "title": doc.get("title", ""),
                "content": doc["content"][:500],  # Truncate for metadata
                **doc.get("metadata", {})
            }
        })

    # Process document chunks
    print("\n--- Processing Document Chunks ---")
    chunks = [chunk["content"] for chunk in data["document_chunks"]]
    chunk_embeddings = embedding_manager.embed_documents(chunks)

    for chunk, emb in zip(data["document_chunks"], chunk_embeddings):
        vectors.append({
            "id": chunk["id"],
            "values": emb.tolist(),
            "metadata": {
                "type": "document_chunk",
                "topic": chunk.get("topic", chunk.get("source", "unknown")),
                "title": chunk.get("title", ""),
                "content": chunk["content"][:500],
                **chunk.get("metadata", {})
            }
        })

    # Process images
    print("\n--- Processing Images ---")
    image_paths = [img["filepath"] for img in data["images"]]
    image_embeddings = embedding_manager.embed_images(image_paths)

    for img, emb in zip(data["images"], image_embeddings):
        vectors.append({
            "id": img["id"],
            "values": emb.tolist(),
            "metadata": {
                "type": "image",
                "diagram_type": img.get("diagram_type", img.get("source", "unknown")),
                "description": img.get("description", img.get("caption", "")),
                "filepath": img["filepath"],
                **img.get("metadata", {})
            }
        })

    print(f"\nTotal vectors prepared: {len(vectors)}")
    return vectors


if __name__ == "__main__":
    # Test embedding generation
    manager = EmbeddingManager()

    # Test text embedding
    test_texts = ["Hello world", "How to configure Kubernetes?"]
    embeddings = manager.embed_texts(test_texts, show_progress=False)
    print(f"Text embeddings shape: {embeddings.shape}")

    # Test query embedding
    query_emb = manager.embed_query("microservices architecture")
    print(f"Query embedding shape: {query_emb.shape}")
