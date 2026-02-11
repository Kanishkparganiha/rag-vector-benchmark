"""Pinecone vector database client for RAG benchmark."""

import time
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from pinecone import Pinecone, ServerlessSpec

import config


class PineconeClient:
    """Client for Pinecone vector database operations."""

    def __init__(self):
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = None
        self.index_name = config.PINECONE_INDEX_NAME

    def create_index(self, dimension: int = config.EMBEDDING_DIMENSION, recreate: bool = False) -> None:
        """Create Pinecone index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name in existing_indexes:
            if recreate:
                print(f"Deleting existing index: {self.index_name}")
                self.pc.delete_index(self.index_name)
                time.sleep(5)  # Wait for deletion
            else:
                print(f"Index {self.index_name} already exists")
                self.index = self.pc.Index(self.index_name)
                return

        print(f"Creating index: {self.index_name} with dimension {dimension}")
        self.pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=config.PINECONE_ENVIRONMENT
            )
        )

        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while True:
            try:
                desc = self.pc.describe_index(self.index_name)
                if desc.status.ready:
                    break
            except Exception:
                pass
            time.sleep(2)

        self.index = self.pc.Index(self.index_name)
        print("Index is ready!")

    def connect(self) -> None:
        """Connect to existing index."""
        self.index = self.pc.Index(self.index_name)
        stats = self.index.describe_index_stats()
        print(f"Connected to index: {self.index_name}")
        print(f"Total vectors: {stats.total_vector_count}")

    def upsert_vectors(
        self,
        vectors: List[Dict],
        batch_size: int = 100,
        namespace: str = ""
    ) -> Dict[str, float]:
        """Upsert vectors in batches and return timing metrics."""
        if self.index is None:
            raise ValueError("Index not connected. Call create_index() or connect() first.")

        metrics = {
            "total_vectors": len(vectors),
            "batch_size": batch_size,
            "total_time": 0,
            "batch_times": []
        }

        start_total = time.time()

        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting vectors"):
            batch = vectors[i:i + batch_size]

            # Format for Pinecone
            upsert_data = [
                {
                    "id": v["id"],
                    "values": v["values"],
                    "metadata": v.get("metadata", {})
                }
                for v in batch
            ]

            batch_start = time.time()
            self.index.upsert(vectors=upsert_data, namespace=namespace)
            batch_time = time.time() - batch_start

            metrics["batch_times"].append(batch_time)

        metrics["total_time"] = time.time() - start_total
        metrics["avg_batch_time"] = sum(metrics["batch_times"]) / len(metrics["batch_times"])
        metrics["vectors_per_second"] = len(vectors) / metrics["total_time"]

        print(f"\nUpsert complete!")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Vectors/second: {metrics['vectors_per_second']:.2f}")

        return metrics

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: str = "",
        filter_dict: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> Tuple[List[Dict], float]:
        """Query the index and return results with timing."""
        if self.index is None:
            raise ValueError("Index not connected.")

        start = time.time()

        results = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=include_metadata
        )

        query_time = time.time() - start

        matches = []
        for match in results.matches:
            matches.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else {}
            })

        return matches, query_time

    def query_by_type(
        self,
        vector: List[float],
        content_type: str,
        top_k: int = 10,
        namespace: str = ""
    ) -> Tuple[List[Dict], float]:
        """Query for specific content type (text, document_chunk, image)."""
        return self.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter_dict={"type": {"$eq": content_type}}
        )

    def hybrid_query(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: str = ""
    ) -> Dict[str, List[Dict]]:
        """Query across all content types and return organized results."""
        results = {
            "text": [],
            "document_chunk": [],
            "image": [],
            "all": []
        }

        # Query without filter for all results
        all_matches, _ = self.query(vector, top_k=top_k * 3, namespace=namespace)
        results["all"] = all_matches[:top_k]

        # Organize by type
        for match in all_matches:
            content_type = match.get("metadata", {}).get("type", "unknown")
            if content_type in results and len(results[content_type]) < top_k:
                results[content_type].append(match)

        return results

    def delete_all(self, namespace: str = "") -> None:
        """Delete all vectors in the namespace."""
        if self.index is None:
            raise ValueError("Index not connected.")

        print("Deleting all vectors...")
        self.index.delete(delete_all=True, namespace=namespace)
        print("All vectors deleted.")

    def get_stats(self) -> Dict:
        """Get index statistics."""
        if self.index is None:
            raise ValueError("Index not connected.")

        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {}
        }


if __name__ == "__main__":
    # Test connection
    client = PineconeClient()

    try:
        client.connect()
        stats = client.get_stats()
        print(f"Index stats: {stats}")
    except Exception as e:
        print(f"Could not connect: {e}")
        print("Run with valid PINECONE_API_KEY to test.")
