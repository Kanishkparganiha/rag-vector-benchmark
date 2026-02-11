"""Generate sample data for RAG benchmark testing."""

import os
import random
from typing import List, Dict, Tuple
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

import config

fake = Faker()

# Tech documentation topics for realistic data
TECH_TOPICS = [
    "API Gateway", "Microservices", "Kubernetes", "Docker", "REST API",
    "GraphQL", "Database Sharding", "Load Balancing", "Caching", "Redis",
    "PostgreSQL", "MongoDB", "Elasticsearch", "Message Queue", "Kafka",
    "RabbitMQ", "Authentication", "OAuth", "JWT", "SSL/TLS",
    "CI/CD Pipeline", "GitHub Actions", "Terraform", "AWS Lambda", "S3",
    "CloudFront", "VPC", "Networking", "DNS", "CDN",
    "Machine Learning", "Neural Networks", "PyTorch", "TensorFlow", "MLOps",
    "Data Pipeline", "ETL", "Spark", "Airflow", "Prometheus",
    "Grafana", "Logging", "Monitoring", "Alerting", "SLO/SLA",
    "Rate Limiting", "Circuit Breaker", "Retry Logic", "Idempotency", "ACID"
]

DIAGRAM_TYPES = [
    "architecture", "flowchart", "sequence", "component", "deployment",
    "network", "database", "api", "pipeline", "infrastructure"
]


def generate_tech_paragraph(topic: str) -> str:
    """Generate a realistic tech documentation paragraph."""
    templates = [
        f"The {topic} component is essential for {fake.bs()}. It enables teams to {fake.catch_phrase().lower()} while maintaining system reliability. Implementation requires careful consideration of {random.choice(TECH_TOPICS)} integration.",
        f"When configuring {topic}, ensure that all dependencies are properly initialized. The recommended approach involves {fake.bs().lower()} with proper error handling. Monitor {random.choice(TECH_TOPICS)} metrics for optimal performance.",
        f"Best practices for {topic} include implementing proper {random.choice(TECH_TOPICS)} strategies. This reduces latency by up to {random.randint(20, 80)}% and improves throughput significantly. Always test in staging before production deployment.",
        f"{topic} architecture follows the principle of {fake.catch_phrase().lower()}. Key components include {random.choice(TECH_TOPICS)} and {random.choice(TECH_TOPICS)}. Regular audits ensure compliance with security standards.",
        f"Troubleshooting {topic} issues requires understanding the underlying {random.choice(TECH_TOPICS)} mechanisms. Common problems include timeout errors, connection pooling issues, and {random.choice(TECH_TOPICS)} misconfiguration.",
    ]
    return random.choice(templates)


def generate_text_documents(num_docs: int = config.NUM_TEXT_DOCUMENTS) -> List[Dict]:
    """Generate sample text documents (short snippets)."""
    print(f"Generating {num_docs} text documents...")
    documents = []

    for i in tqdm(range(num_docs)):
        topic = random.choice(TECH_TOPICS)
        doc = {
            "id": f"text_{i:04d}",
            "type": "text",
            "topic": topic,
            "content": generate_tech_paragraph(topic),
            "metadata": {
                "author": fake.name(),
                "created_at": fake.date_this_year().isoformat(),
                "category": random.choice(["tutorial", "reference", "guide", "troubleshooting"]),
                "difficulty": random.choice(["beginner", "intermediate", "advanced"])
            }
        }
        documents.append(doc)

    return documents


def generate_document_chunks(num_chunks: int = config.NUM_DOCUMENT_CHUNKS) -> List[Dict]:
    """Generate document chunks (longer, multi-paragraph content)."""
    print(f"Generating {num_chunks} document chunks...")
    chunks = []

    for i in tqdm(range(num_chunks)):
        topic = random.choice(TECH_TOPICS)
        # Generate 2-4 paragraphs per chunk
        num_paragraphs = random.randint(2, 4)
        content = "\n\n".join([generate_tech_paragraph(topic) for _ in range(num_paragraphs)])

        chunk = {
            "id": f"chunk_{i:04d}",
            "type": "document_chunk",
            "topic": topic,
            "content": content,
            "metadata": {
                "source_doc": f"doc_{i // 10:03d}.md",
                "chunk_index": i % 10,
                "total_chunks": 10,
                "word_count": len(content.split()),
                "section": random.choice(["introduction", "implementation", "configuration", "examples", "troubleshooting"])
            }
        }
        chunks.append(chunk)

    return chunks


def generate_diagram_image(diagram_type: str, index: int, size: Tuple[int, int] = (400, 300)) -> Image.Image:
    """Generate a simple diagram image with shapes and text."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    # Draw random shapes to simulate a diagram
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

    # Draw some rectangles (boxes)
    num_boxes = random.randint(3, 6)
    for _ in range(num_boxes):
        x1 = random.randint(20, size[0] - 100)
        y1 = random.randint(40, size[1] - 80)
        x2 = x1 + random.randint(60, 100)
        y2 = y1 + random.randint(30, 50)
        color = random.choice(colors)
        draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=2)

        # Add small text in box
        text = random.choice(TECH_TOPICS)[:10]
        draw.text((x1 + 5, y1 + 10), text, fill='white')

    # Draw some lines (connections)
    for _ in range(random.randint(2, 5)):
        x1 = random.randint(50, size[0] - 50)
        y1 = random.randint(50, size[1] - 50)
        x2 = random.randint(50, size[0] - 50)
        y2 = random.randint(50, size[1] - 50)
        draw.line([x1, y1, x2, y2], fill='gray', width=2)

    # Add title
    title = f"{diagram_type.title()} Diagram #{index + 1}"
    draw.text((10, 10), title, fill='black')

    return img


def generate_images(num_images: int = config.NUM_IMAGES, output_dir: str = config.IMAGES_DIR) -> List[Dict]:
    """Generate sample diagram images."""
    print(f"Generating {num_images} sample images...")
    os.makedirs(output_dir, exist_ok=True)

    images = []
    for i in tqdm(range(num_images)):
        diagram_type = random.choice(DIAGRAM_TYPES)
        img = generate_diagram_image(diagram_type, i)

        filename = f"{diagram_type}_{i:03d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)

        image_info = {
            "id": f"img_{i:04d}",
            "type": "image",
            "filepath": filepath,
            "diagram_type": diagram_type,
            "description": f"A {diagram_type} diagram showing {random.choice(TECH_TOPICS)} architecture",
            "metadata": {
                "width": img.size[0],
                "height": img.size[1],
                "format": "PNG",
                "topic": random.choice(TECH_TOPICS)
            }
        }
        images.append(image_info)

    return images


def generate_all_data() -> Dict[str, List[Dict]]:
    """Generate all sample data."""
    print("=" * 50)
    print("Generating Sample Data for RAG Benchmark")
    print("=" * 50)

    data = {
        "text_documents": generate_text_documents(),
        "document_chunks": generate_document_chunks(),
        "images": generate_images()
    }

    print("\nData generation complete!")
    print(f"  - Text documents: {len(data['text_documents'])}")
    print(f"  - Document chunks: {len(data['document_chunks'])}")
    print(f"  - Images: {len(data['images'])}")

    return data


def get_sample_queries() -> List[Dict]:
    """Generate sample queries for testing."""
    queries = []

    # Text queries
    for topic in random.sample(TECH_TOPICS, 20):
        queries.append({
            "query": f"How do I configure {topic}?",
            "type": "text",
            "expected_topic": topic
        })
        queries.append({
            "query": f"Best practices for {topic} implementation",
            "type": "text",
            "expected_topic": topic
        })

    # Document queries
    queries.extend([
        {"query": "Explain microservices architecture patterns", "type": "document"},
        {"query": "Database optimization techniques", "type": "document"},
        {"query": "Kubernetes deployment strategies", "type": "document"},
        {"query": "API security best practices", "type": "document"},
        {"query": "Monitoring and alerting setup", "type": "document"},
    ])

    # Image queries
    for diagram_type in DIAGRAM_TYPES:
        queries.append({
            "query": f"Show me {diagram_type} diagrams",
            "type": "image",
            "expected_diagram": diagram_type
        })

    return queries


if __name__ == "__main__":
    data = generate_all_data()
    queries = get_sample_queries()
    print(f"\nGenerated {len(queries)} sample queries for testing.")
