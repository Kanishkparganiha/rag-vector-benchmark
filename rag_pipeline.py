"""RAG Pipeline for multi-modal retrieval and generation."""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass

import config
from embeddings import EmbeddingManager
from pinecone_client import PineconeClient


@dataclass
class RAGResult:
    """Result from RAG query."""
    query: str
    retrieved_docs: List[Dict]
    generated_response: Optional[str] = None
    query_time: float = 0.0
    generation_time: float = 0.0


class RAGPipeline:
    """Multi-modal RAG pipeline supporting text, documents, and images."""

    def __init__(self, use_gpu: bool = False):
        self.embedding_manager = EmbeddingManager(use_gpu=use_gpu)
        self.pinecone_client = PineconeClient()
        self._openai_client = None

    def initialize(self, recreate_index: bool = False) -> None:
        """Initialize the pipeline and connect to Pinecone."""
        self.pinecone_client.create_index(recreate=recreate_index)

    def connect(self) -> None:
        """Connect to existing Pinecone index."""
        self.pinecone_client.connect()

    @property
    def openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI()
            except Exception as e:
                print(f"OpenAI client not available: {e}")
                self._openai_client = "unavailable"
        return self._openai_client

    def ingest(self, data: Dict[str, List[Dict]], batch_size: int = 100) -> Dict:
        """Ingest all data types into Pinecone."""
        from embeddings import prepare_vectors_for_pinecone

        print("\n" + "=" * 50)
        print("Starting Data Ingestion")
        print("=" * 50)

        vectors = prepare_vectors_for_pinecone(data, self.embedding_manager)
        metrics = self.pinecone_client.upsert_vectors(vectors, batch_size=batch_size)

        return metrics

    def search_text(
        self,
        query: str,
        top_k: int = 10,
        content_type: Optional[str] = None
    ) -> RAGResult:
        """Search for text documents."""
        query_vector = self.embedding_manager.embed_query(query)

        if content_type:
            matches, query_time = self.pinecone_client.query_by_type(
                query_vector.tolist(),
                content_type=content_type,
                top_k=top_k
            )
        else:
            matches, query_time = self.pinecone_client.query(
                query_vector.tolist(),
                top_k=top_k
            )

        return RAGResult(
            query=query,
            retrieved_docs=matches,
            query_time=query_time
        )

    def search_images(self, query: str, top_k: int = 10) -> RAGResult:
        """Search for images using CLIP-based text query."""
        query_vector = self.embedding_manager.embed_image_query(query)

        matches, query_time = self.pinecone_client.query_by_type(
            query_vector.tolist(),
            content_type="image",
            top_k=top_k
        )

        return RAGResult(
            query=query,
            retrieved_docs=matches,
            query_time=query_time
        )

    def hybrid_search(self, query: str, top_k: int = 10) -> Dict[str, RAGResult]:
        """Search across all content types."""
        query_vector = self.embedding_manager.embed_query(query)

        results = self.pinecone_client.hybrid_query(
            query_vector.tolist(),
            top_k=top_k
        )

        return {
            content_type: RAGResult(
                query=query,
                retrieved_docs=docs,
                query_time=0.0
            )
            for content_type, docs in results.items()
        }

    def generate_response(
        self,
        query: str,
        context_docs: List[Dict],
        max_tokens: int = 500
    ) -> str:
        """Generate a response using retrieved context (requires OpenAI API key)."""
        if self.openai_client == "unavailable":
            return self._generate_mock_response(query, context_docs)

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs[:5], 1):  # Use top 5
            content = doc.get("metadata", {}).get("content", "")
            doc_type = doc.get("metadata", {}).get("type", "unknown")
            context_parts.append(f"[{i}] ({doc_type}) {content[:300]}...")

        context = "\n\n".join(context_parts)

        prompt = f"""Based on the following retrieved documents, answer the user's question.

Retrieved Documents:
{context}

User Question: {query}

Answer concisely based on the retrieved information:"""

        try:
            import time
            start = time.time()

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful technical assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            generation_time = time.time() - start
            return response.choices[0].message.content

        except Exception as e:
            return f"Generation error: {e}\n\n{self._generate_mock_response(query, context_docs)}"

    def _generate_mock_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate a mock response when OpenAI is not available."""
        if not context_docs:
            return "No relevant documents found for your query."

        top_doc = context_docs[0]
        content = top_doc.get("metadata", {}).get("content", "No content available")
        score = top_doc.get("score", 0)
        doc_type = top_doc.get("metadata", {}).get("type", "unknown")

        response = f"""[Mock Response - OpenAI API not configured]

Based on your query: "{query}"

Most relevant {doc_type} (similarity: {score:.3f}):
{content[:500]}

To enable AI-generated responses, set your OPENAI_API_KEY in .env file."""

        return response

    def rag_query(
        self,
        query: str,
        top_k: int = 5,
        generate: bool = True
    ) -> RAGResult:
        """Full RAG query: retrieve and generate."""
        import time

        # Retrieve
        result = self.search_text(query, top_k=top_k)

        # Generate
        if generate and result.retrieved_docs:
            start = time.time()
            result.generated_response = self.generate_response(query, result.retrieved_docs)
            result.generation_time = time.time() - start

        return result


def demo_rag_pipeline():
    """Demonstrate the RAG pipeline with sample queries."""
    print("\n" + "=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)

    pipeline = RAGPipeline()
    pipeline.connect()

    # Sample queries
    queries = [
        "How do I configure Kubernetes for high availability?",
        "What are best practices for API security?",
        "Show me architecture diagrams",
    ]

    for query in queries:
        print(f"\n{'─' * 50}")
        print(f"Query: {query}")
        print('─' * 50)

        result = pipeline.rag_query(query, top_k=3, generate=True)

        print(f"Query time: {result.query_time * 1000:.2f}ms")
        print(f"Documents retrieved: {len(result.retrieved_docs)}")

        for i, doc in enumerate(result.retrieved_docs, 1):
            doc_type = doc.get("metadata", {}).get("type", "?")
            score = doc.get("score", 0)
            print(f"  [{i}] {doc_type} - score: {score:.3f}")

        if result.generated_response:
            print(f"\nGenerated Response:")
            print(result.generated_response[:500])


if __name__ == "__main__":
    demo_rag_pipeline()
