"""Stress testing scenarios for Pinecone RAG benchmark."""

import time
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

import config
from embeddings import EmbeddingManager
from pinecone_client import PineconeClient
from data_generator import get_sample_queries, TECH_TOPICS


@dataclass
class StressTestResult:
    """Results from a stress test."""
    test_name: str
    total_requests: int
    total_time: float
    successful_requests: int
    failed_requests: int
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        return np.mean(self.latencies) if self.latencies else 0

    @property
    def p50_latency(self) -> float:
        return np.percentile(self.latencies, 50) if self.latencies else 0

    @property
    def p95_latency(self) -> float:
        return np.percentile(self.latencies, 95) if self.latencies else 0

    @property
    def p99_latency(self) -> float:
        return np.percentile(self.latencies, 99) if self.latencies else 0

    @property
    def throughput(self) -> float:
        return self.successful_requests / self.total_time if self.total_time > 0 else 0

    @property
    def error_rate(self) -> float:
        return self.failed_requests / self.total_requests if self.total_requests > 0 else 0

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_time_s": round(self.total_time, 3),
            "avg_latency_ms": round(self.avg_latency * 1000, 2),
            "p50_latency_ms": round(self.p50_latency * 1000, 2),
            "p95_latency_ms": round(self.p95_latency * 1000, 2),
            "p99_latency_ms": round(self.p99_latency * 1000, 2),
            "throughput_qps": round(self.throughput, 2),
            "error_rate": round(self.error_rate, 4)
        }


class StressTestSuite:
    """Suite of stress tests for Pinecone."""

    def __init__(self, use_gpu: bool = False):
        self.embedding_manager = EmbeddingManager(use_gpu=use_gpu)
        self.pinecone_client = PineconeClient()
        self.results: List[StressTestResult] = []

    def connect(self):
        """Connect to Pinecone."""
        self.pinecone_client.connect()

    def _generate_query_vector(self) -> List[float]:
        """Generate a random query vector."""
        topic = random.choice(TECH_TOPICS)
        query = f"How to implement {topic}?"
        return self.embedding_manager.embed_query(query).tolist()

    def test_concurrent_queries(
        self,
        num_concurrent: int = 10,
        num_iterations: int = 5,
        top_k: int = 10
    ) -> StressTestResult:
        """Test concurrent query performance."""
        print(f"\n{'=' * 50}")
        print(f"Concurrent Query Test: {num_concurrent} concurrent queries")
        print(f"{'=' * 50}")

        result = StressTestResult(
            test_name=f"concurrent_queries_{num_concurrent}",
            total_requests=num_concurrent * num_iterations,
            total_time=0,
            successful_requests=0,
            failed_requests=0
        )

        # Pre-generate query vectors
        print("Pre-generating query vectors...")
        query_vectors = [self._generate_query_vector() for _ in range(num_concurrent)]

        def execute_query(vector: List[float]) -> tuple:
            try:
                start = time.time()
                matches, _ = self.pinecone_client.query(vector, top_k=top_k)
                latency = time.time() - start
                return True, latency, None
            except Exception as e:
                return False, 0, str(e)

        start_total = time.time()

        for iteration in tqdm(range(num_iterations), desc="Running iterations"):
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(execute_query, v) for v in query_vectors]

                for future in as_completed(futures):
                    success, latency, error = future.result()
                    if success:
                        result.successful_requests += 1
                        result.latencies.append(latency)
                    else:
                        result.failed_requests += 1
                        result.errors.append(error)

        result.total_time = time.time() - start_total

        print(f"  Successful: {result.successful_requests}/{result.total_requests}")
        print(f"  Avg latency: {result.avg_latency * 1000:.2f}ms")
        print(f"  P95 latency: {result.p95_latency * 1000:.2f}ms")
        print(f"  Throughput: {result.throughput:.2f} QPS")

        self.results.append(result)
        return result

    def test_batch_insert_performance(
        self,
        num_vectors: int = 200,
        batch_sizes: List[int] = None
    ) -> List[StressTestResult]:
        """Test batch insert performance with different batch sizes."""
        if batch_sizes is None:
            batch_sizes = config.BATCH_SIZES

        print(f"\n{'=' * 50}")
        print(f"Batch Insert Performance Test")
        print(f"{'=' * 50}")

        results = []

        # Generate test vectors
        print(f"Generating {num_vectors} test vectors...")
        test_vectors = []
        for i in range(num_vectors):
            topic = random.choice(TECH_TOPICS)
            content = f"Test document about {topic} for stress testing."
            embedding = self.embedding_manager.embed_query(content)
            test_vectors.append({
                "id": f"stress_test_{i:05d}",
                "values": embedding.tolist(),
                "metadata": {"type": "stress_test", "topic": topic}
            })

        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")

            result = StressTestResult(
                test_name=f"batch_insert_{batch_size}",
                total_requests=num_vectors,
                total_time=0,
                successful_requests=0,
                failed_requests=0
            )

            start = time.time()

            for i in range(0, num_vectors, batch_size):
                batch = test_vectors[i:i + batch_size]
                batch_start = time.time()

                try:
                    upsert_data = [
                        {"id": v["id"], "values": v["values"], "metadata": v["metadata"]}
                        for v in batch
                    ]
                    self.pinecone_client.index.upsert(vectors=upsert_data)
                    result.successful_requests += len(batch)
                    result.latencies.append(time.time() - batch_start)
                except Exception as e:
                    result.failed_requests += len(batch)
                    result.errors.append(str(e))

            result.total_time = time.time() - start

            print(f"  Total time: {result.total_time:.2f}s")
            print(f"  Vectors/second: {num_vectors / result.total_time:.2f}")

            results.append(result)
            self.results.append(result)

        # Cleanup stress test vectors
        print("\nCleaning up stress test vectors...")
        try:
            ids_to_delete = [f"stress_test_{i:05d}" for i in range(num_vectors)]
            self.pinecone_client.index.delete(ids=ids_to_delete)
        except Exception as e:
            print(f"Cleanup warning: {e}")

        return results

    def test_query_latency_under_load(
        self,
        duration_seconds: int = 30,
        queries_per_second: int = 5
    ) -> StressTestResult:
        """Test query latency under sustained load."""
        print(f"\n{'=' * 50}")
        print(f"Query Latency Under Load Test ({duration_seconds}s @ {queries_per_second} QPS)")
        print(f"{'=' * 50}")

        result = StressTestResult(
            test_name=f"latency_under_load_{queries_per_second}qps",
            total_requests=0,
            total_time=duration_seconds,
            successful_requests=0,
            failed_requests=0
        )

        # Pre-generate query vectors
        query_vectors = [self._generate_query_vector() for _ in range(100)]

        interval = 1.0 / queries_per_second
        start_time = time.time()
        query_index = 0

        with tqdm(total=duration_seconds, desc="Running load test", unit="s") as pbar:
            last_update = start_time
            while time.time() - start_time < duration_seconds:
                query_start = time.time()

                try:
                    vector = query_vectors[query_index % len(query_vectors)]
                    matches, latency = self.pinecone_client.query(vector, top_k=10)
                    result.successful_requests += 1
                    result.latencies.append(latency)
                except Exception as e:
                    result.failed_requests += 1
                    result.errors.append(str(e))

                result.total_requests += 1
                query_index += 1

                # Maintain query rate
                elapsed = time.time() - query_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)

                # Update progress bar
                current_time = time.time()
                if current_time - last_update >= 1:
                    pbar.update(int(current_time - last_update))
                    last_update = current_time

        print(f"  Total queries: {result.total_requests}")
        print(f"  Avg latency: {result.avg_latency * 1000:.2f}ms")
        print(f"  P95 latency: {result.p95_latency * 1000:.2f}ms")
        print(f"  Actual QPS: {result.throughput:.2f}")

        self.results.append(result)
        return result

    def test_different_k_values(
        self,
        k_values: List[int] = None,
        num_queries: int = 50
    ) -> List[StressTestResult]:
        """Test retrieval performance with different k values."""
        if k_values is None:
            k_values = config.TOP_K_VALUES

        print(f"\n{'=' * 50}")
        print(f"Different K Values Test")
        print(f"{'=' * 50}")

        results = []

        # Pre-generate query vectors
        query_vectors = [self._generate_query_vector() for _ in range(num_queries)]

        for k in k_values:
            print(f"\nTesting top_k={k}")

            result = StressTestResult(
                test_name=f"top_k_{k}",
                total_requests=num_queries,
                total_time=0,
                successful_requests=0,
                failed_requests=0
            )

            start = time.time()

            for vector in tqdm(query_vectors, desc=f"Querying with k={k}"):
                try:
                    query_start = time.time()
                    matches, _ = self.pinecone_client.query(vector, top_k=k)
                    result.latencies.append(time.time() - query_start)
                    result.successful_requests += 1
                except Exception as e:
                    result.failed_requests += 1
                    result.errors.append(str(e))

            result.total_time = time.time() - start

            print(f"  Avg latency: {result.avg_latency * 1000:.2f}ms")
            print(f"  P95 latency: {result.p95_latency * 1000:.2f}ms")

            results.append(result)
            self.results.append(result)

        return results

    def run_all_tests(self) -> List[StressTestResult]:
        """Run all stress tests."""
        print("\n" + "=" * 60)
        print("RUNNING FULL STRESS TEST SUITE")
        print("=" * 60)

        self.results = []

        # 1. Concurrent queries tests
        for concurrency in config.CONCURRENT_QUERIES:
            self.test_concurrent_queries(
                num_concurrent=concurrency,
                num_iterations=config.NUM_STRESS_ITERATIONS
            )

        # 2. Batch insert tests
        self.test_batch_insert_performance()

        # 3. Different k values
        self.test_different_k_values()

        # 4. Latency under load (shorter duration for demo)
        self.test_query_latency_under_load(duration_seconds=15, queries_per_second=5)

        print("\n" + "=" * 60)
        print("STRESS TEST SUITE COMPLETE")
        print(f"Total tests run: {len(self.results)}")
        print("=" * 60)

        return self.results

    def get_summary(self) -> Dict:
        """Get summary of all test results."""
        return {
            "total_tests": len(self.results),
            "tests": [r.to_dict() for r in self.results]
        }


if __name__ == "__main__":
    suite = StressTestSuite()
    suite.connect()

    # Run a quick test
    suite.test_concurrent_queries(num_concurrent=5, num_iterations=2)
    suite.test_different_k_values(k_values=[5, 10], num_queries=10)

    print("\nSummary:")
    for result in suite.results:
        print(f"  {result.test_name}: {result.avg_latency * 1000:.2f}ms avg, {result.throughput:.2f} QPS")
