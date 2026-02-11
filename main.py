#!/usr/bin/env python3
"""
RAG Vector Benchmark - Main Entry Point

A comprehensive stress testing tool for Pinecone vector database
with multi-modal RAG support (text, documents, images).

Usage:
    python main.py                  # Run full benchmark
    python main.py --ingest-only    # Only ingest data
    python main.py --test-only      # Only run stress tests
    python main.py --demo           # Run quick demo
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_generator import generate_all_data, get_sample_queries
from data_loader import PublicDataLoader, DATASETS
from embeddings import EmbeddingManager
from pinecone_client import PineconeClient
from rag_pipeline import RAGPipeline
from stress_test import StressTestSuite
from visualizations import MetricsVisualizer


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     RAG VECTOR BENCHMARK                                 ║
    ║     Pinecone Stress Testing Tool                         ║
    ║                                                          ║
    ║     Multi-modal RAG: Text | Documents | Images           ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """Check if environment is properly configured."""
    print("\nChecking environment...")

    issues = []

    # Check Pinecone API key
    if config.PINECONE_API_KEY == "your-api-key-here":
        issues.append("PINECONE_API_KEY not set in .env file")

    # Check if .env file exists
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            print("  Note: Copy .env.example to .env and add your API keys")

    if issues:
        print("\n⚠️  Configuration issues found:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n  Please fix these issues before running the benchmark.")
        return False

    print("  ✓ Environment configured correctly")
    return True


def run_data_ingestion(
    recreate_index: bool = False,
    use_public_data: bool = True,
    sources: list = None
) -> dict:
    """Run data generation and ingestion pipeline."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA LOADING & INGESTION")
    print("=" * 60)

    # Load data from public sources or generate synthetic
    if use_public_data:
        loader = PublicDataLoader()
        if sources is None:
            sources = ["wikipedia", "squad"]  # Default sources
        data = loader.load_all(
            num_texts=config.NUM_TEXT_DOCUMENTS,
            num_chunks=config.NUM_DOCUMENT_CHUNKS,
            num_images=config.NUM_IMAGES,
            sources=sources
        )
    else:
        print("Using synthetic data generation...")
        data = generate_all_data()

    # Initialize RAG pipeline
    print("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline()
    pipeline.initialize(recreate_index=recreate_index)

    # Ingest data
    print("\nIngesting data into Pinecone...")
    ingest_metrics = pipeline.ingest(data, batch_size=100)

    # Wait for vectors to be indexed
    print("\nWaiting for vectors to be indexed...")
    time.sleep(10)

    # Verify ingestion
    stats = pipeline.pinecone_client.get_stats()
    print(f"\nIngestion complete!")
    print(f"  Total vectors in index: {stats['total_vectors']}")

    return {
        "data": data,
        "ingest_metrics": ingest_metrics,
        "index_stats": stats
    }


def run_stress_tests() -> list:
    """Run stress testing suite."""
    print("\n" + "=" * 60)
    print("PHASE 2: STRESS TESTING")
    print("=" * 60)

    suite = StressTestSuite()
    suite.connect()

    results = suite.run_all_tests()
    return results


def run_rag_demo():
    """Run RAG demonstration queries."""
    print("\n" + "=" * 60)
    print("PHASE 3: RAG DEMONSTRATION")
    print("=" * 60)

    pipeline = RAGPipeline()
    pipeline.connect()

    demo_queries = [
        "How do I configure Kubernetes for production?",
        "What are the best practices for API authentication?",
        "Explain database sharding strategies",
        "Show me architecture diagrams for microservices",
    ]

    print("\nRunning demo queries...")
    for query in demo_queries:
        print(f"\n{'─' * 50}")
        print(f"Query: {query}")
        print('─' * 50)

        result = pipeline.rag_query(query, top_k=3, generate=True)

        print(f"Query time: {result.query_time * 1000:.2f}ms")
        print(f"Retrieved {len(result.retrieved_docs)} documents:")

        for i, doc in enumerate(result.retrieved_docs, 1):
            doc_type = doc.get("metadata", {}).get("type", "?")
            topic = doc.get("metadata", {}).get("topic", "?")
            score = doc.get("score", 0)
            print(f"  [{i}] {doc_type} | {topic} | score: {score:.3f}")

        if result.generated_response:
            print(f"\nGenerated Response:")
            print("-" * 40)
            # Truncate for display
            response = result.generated_response
            if len(response) > 500:
                response = response[:500] + "..."
            print(response)


def generate_visualizations(results: list):
    """Generate all visualizations and reports."""
    print("\n" + "=" * 60)
    print("PHASE 4: VISUALIZATION & REPORTING")
    print("=" * 60)

    visualizer = MetricsVisualizer()
    files = visualizer.generate_all_visualizations(results)

    print(f"\n✓ All visualizations saved to {config.OUTPUT_DIR}/")
    return files


def run_quick_demo(use_public_data: bool = True):
    """Run a quick demo without full stress testing."""
    print_banner()

    if not check_environment():
        return

    print("\n🚀 Running Quick Demo Mode")
    print("This will load sample data and run basic tests.\n")

    # Load minimal data
    if use_public_data:
        print("Loading public datasets (Wikipedia + SQuAD)...")
        loader = PublicDataLoader()
        data = loader.load_all(
            num_texts=100,
            num_chunks=50,
            num_images=10,
            sources=["wikipedia", "squad"]
        )
    else:
        print("Generating synthetic data...")
        from data_generator import (
            generate_text_documents,
            generate_document_chunks,
            generate_images
        )
        data = {
            "text_documents": generate_text_documents(50),
            "document_chunks": generate_document_chunks(20),
            "images": generate_images(10)
        }

    # Initialize and ingest
    pipeline = RAGPipeline()

    try:
        pipeline.connect()
        stats = pipeline.pinecone_client.get_stats()
        if stats['total_vectors'] > 0:
            print(f"Using existing index with {stats['total_vectors']} vectors")
        else:
            pipeline.initialize(recreate_index=True)
            pipeline.ingest(data, batch_size=50)
            time.sleep(5)
    except Exception:
        print("Creating new index...")
        pipeline.initialize(recreate_index=True)
        pipeline.ingest(data, batch_size=50)
        time.sleep(5)

    # Quick stress test
    print("\nRunning quick stress tests...")
    suite = StressTestSuite()
    suite.connect()

    suite.test_concurrent_queries(num_concurrent=5, num_iterations=2)
    suite.test_different_k_values(k_values=[5, 10], num_queries=10)

    # Visualizations
    visualizer = MetricsVisualizer()
    visualizer.generate_all_visualizations(suite.results)

    # Demo queries
    run_rag_demo()

    print("\n" + "=" * 60)
    print("✓ Quick Demo Complete!")
    print(f"Check {config.OUTPUT_DIR}/ for results and visualizations")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Vector Benchmark - Pinecone Stress Testing Tool"
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only run data ingestion"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run stress tests (assumes data already ingested)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demo with minimal data"
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate Pinecone index"
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of public datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="wikipedia,squad",
        help="Comma-separated list of datasets to use. Options: " +
             "wikipedia, arxiv, squad, msmarco, news, stackoverflow, " +
             "pubmed, financial, paul_graham, github_readme, coco"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available public datasets and exit"
    )

    args = parser.parse_args()

    # List datasets and exit
    if args.list_datasets:
        print("\nAvailable Public Datasets:")
        print("=" * 60)
        for name, desc in DATASETS.items():
            print(f"  {name:15} - {desc}")
        print("\nUsage: python main.py --datasets wikipedia,arxiv,squad")
        return

    print_banner()

    if not check_environment():
        sys.exit(1)

    # Quick demo mode
    if args.demo:
        run_quick_demo(use_public_data=not args.synthetic)
        return

    start_time = time.time()
    results = []

    # Parse dataset sources
    sources = [s.strip() for s in args.datasets.split(",")]

    try:
        # Ingestion phase
        if not args.test_only:
            ingestion_result = run_data_ingestion(
                recreate_index=args.recreate_index,
                use_public_data=not args.synthetic,
                sources=sources
            )

            if args.ingest_only:
                print("\n✓ Ingestion complete!")
                return

        # Stress testing phase
        if not args.ingest_only:
            results = run_stress_tests()

        # RAG Demo
        run_rag_demo()

        # Visualization phase
        if not args.skip_visualization and results:
            generate_visualizations(results)

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"Total execution time: {total_time / 60:.1f} minutes")
        print(f"Results saved to: {config.OUTPUT_DIR}/")
        print("\nKey files generated:")
        print(f"  - {config.OUTPUT_DIR}/dashboard.html (Interactive dashboard)")
        print(f"  - {config.OUTPUT_DIR}/summary_report.txt (Text report)")
        print(f"  - {config.OUTPUT_DIR}/results.json (Raw data)")
        print(f"  - {config.OUTPUT_DIR}/*.png (Charts)")

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
