"""Load real public datasets for RAG benchmark testing."""

import os
import json
import random
import requests
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from pathlib import Path

import config

# Dataset sources
DATASETS = {
    # Text/Document Sources
    "wikipedia": "Real Wikipedia articles (tech/science topics)",
    "arxiv": "ArXiv ML/AI paper abstracts",
    "msmarco": "MS MARCO passage retrieval dataset",
    "squad": "Stanford Question Answering Dataset contexts",
    "paul_graham": "Paul Graham's essays (popular for RAG demos)",
    "news": "AG News / BBC News articles",
    "stackoverflow": "Stack Overflow Q&A (programming)",
    "pubmed": "PubMed medical/scientific abstracts",
    "github_readme": "GitHub repository READMEs",
    "legal": "Legal documents (contracts, cases)",
    "financial": "Financial news and SEC filings",

    # Image Sources
    "coco": "COCO dataset (real photos with captions)",
    "diagrams": "Technical diagrams from Wikimedia",
    "charts": "Data visualization charts",
}


class PublicDataLoader:
    """Load real public datasets for RAG benchmarking."""

    def __init__(self, data_dir: str = config.DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    def load_wikipedia_articles(self, num_articles: int = 1000) -> List[Dict]:
        """
        Load Wikipedia articles using HuggingFace datasets.
        Uses 'wikipedia' dataset - simple English for manageable size.
        """
        print(f"Loading {num_articles} Wikipedia articles...")

        try:
            from datasets import load_dataset

            # Load simple English Wikipedia (smaller, cleaner)
            dataset = load_dataset(
                "wikipedia",
                "20220301.simple",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            articles = []
            tech_keywords = [
                "software", "computer", "programming", "algorithm", "database",
                "network", "internet", "server", "cloud", "machine learning",
                "artificial intelligence", "data", "system", "technology",
                "engineering", "science", "physics", "mathematics", "chemistry",
                "biology", "medicine", "research", "university", "company"
            ]

            for i, article in enumerate(tqdm(dataset, total=num_articles, desc="Loading Wikipedia")):
                if i >= num_articles * 3:  # Search through more to find relevant ones
                    break

                text = article.get("text", "")
                title = article.get("title", "")

                # Filter for tech/science content (optional, remove for variety)
                # if not any(kw in text.lower() for kw in tech_keywords):
                #     continue

                if len(text) < 200:  # Skip very short articles
                    continue

                # Take first ~500 chars as a document
                content = text[:1000].strip()
                if not content:
                    continue

                articles.append({
                    "id": f"wiki_{len(articles):04d}",
                    "type": "text",
                    "source": "wikipedia",
                    "title": title,
                    "content": content,
                    "metadata": {
                        "source": "wikipedia",
                        "title": title,
                        "url": f"https://simple.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    }
                })

                if len(articles) >= num_articles:
                    break

            print(f"  Loaded {len(articles)} Wikipedia articles")
            return articles

        except ImportError:
            print("  datasets library not installed. Installing...")
            os.system("pip install datasets")
            return self.load_wikipedia_articles(num_articles)
        except Exception as e:
            print(f"  Error loading Wikipedia: {e}")
            print("  Falling back to API method...")
            return self._load_wikipedia_api(num_articles)

    def _load_wikipedia_api(self, num_articles: int) -> List[Dict]:
        """Fallback: Load Wikipedia via REST API."""
        articles = []

        # Tech/science topics to search
        topics = [
            "Machine learning", "Cloud computing", "Kubernetes", "Docker",
            "Python programming", "JavaScript", "Database", "API",
            "Neural network", "Deep learning", "Data science", "DevOps",
            "Microservices", "Linux", "Git", "SQL", "NoSQL", "Redis",
            "Elasticsearch", "Apache Kafka", "REST API", "GraphQL",
            "Computer science", "Software engineering", "Agile",
            "Cybersecurity", "Encryption", "Blockchain", "Web development"
        ]

        for topic in tqdm(topics[:num_articles // 3], desc="Fetching Wikipedia"):
            try:
                # Wikipedia API
                url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    extract = data.get("extract", "")

                    if len(extract) > 100:
                        articles.append({
                            "id": f"wiki_{len(articles):04d}",
                            "type": "text",
                            "source": "wikipedia",
                            "title": data.get("title", topic),
                            "content": extract,
                            "metadata": {
                                "source": "wikipedia",
                                "title": data.get("title", topic),
                                "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                            }
                        })
            except Exception as e:
                continue

        return articles

    def load_arxiv_abstracts(self, num_papers: int = 200) -> List[Dict]:
        """
        Load ArXiv paper abstracts (ML/AI focused).
        Uses HuggingFace's arxiv dataset.
        """
        print(f"Loading {num_papers} ArXiv abstracts...")

        try:
            from datasets import load_dataset

            # ArXiv dataset from HuggingFace
            dataset = load_dataset(
                "ccdv/arxiv-classification",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            papers = []
            for i, paper in enumerate(tqdm(dataset, total=num_papers, desc="Loading ArXiv")):
                if i >= num_papers:
                    break

                abstract = paper.get("text", "")
                if len(abstract) < 100:
                    continue

                papers.append({
                    "id": f"arxiv_{len(papers):04d}",
                    "type": "document_chunk",
                    "source": "arxiv",
                    "content": abstract[:1500],
                    "metadata": {
                        "source": "arxiv",
                        "category": paper.get("label", "unknown"),
                    }
                })

            print(f"  Loaded {len(papers)} ArXiv abstracts")
            return papers

        except Exception as e:
            print(f"  Error loading ArXiv: {e}")
            return self._load_arxiv_api(num_papers)

    def _load_arxiv_api(self, num_papers: int) -> List[Dict]:
        """Fallback: Load ArXiv via API."""
        import urllib.request
        import xml.etree.ElementTree as ET

        papers = []
        categories = ["cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.DB", "cs.SE"]

        for cat in categories[:3]:
            try:
                url = f"http://export.arxiv.org/api/query?search_query=cat:{cat}&start=0&max_results={num_papers // 3}"
                response = urllib.request.urlopen(url, timeout=30)
                data = response.read().decode("utf-8")

                root = ET.fromstring(data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                for entry in root.findall("atom:entry", ns):
                    title = entry.find("atom:title", ns)
                    summary = entry.find("atom:summary", ns)
                    arxiv_id = entry.find("atom:id", ns)

                    if summary is not None and title is not None:
                        papers.append({
                            "id": f"arxiv_{len(papers):04d}",
                            "type": "document_chunk",
                            "source": "arxiv",
                            "title": title.text.strip().replace("\n", " "),
                            "content": summary.text.strip().replace("\n", " "),
                            "metadata": {
                                "source": "arxiv",
                                "category": cat,
                                "arxiv_id": arxiv_id.text if arxiv_id is not None else "",
                            }
                        })
            except Exception as e:
                print(f"  ArXiv API error for {cat}: {e}")
                continue

        return papers[:num_papers]

    def load_msmarco_passages(self, num_passages: int = 500) -> List[Dict]:
        """
        Load MS MARCO passage retrieval dataset.
        High-quality web passages used for search/retrieval research.
        """
        print(f"Loading {num_passages} MS MARCO passages...")

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "ms_marco",
                "v1.1",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            passages = []
            seen_passages = set()

            for item in tqdm(dataset, total=num_passages * 2, desc="Loading MS MARCO"):
                if len(passages) >= num_passages:
                    break

                # MS MARCO has passages in a list
                passage_list = item.get("passages", {})
                passage_texts = passage_list.get("passage_text", [])

                for text in passage_texts:
                    if text in seen_passages or len(text) < 100:
                        continue

                    seen_passages.add(text)
                    passages.append({
                        "id": f"msmarco_{len(passages):04d}",
                        "type": "document_chunk",
                        "source": "msmarco",
                        "content": text[:1000],
                        "query": item.get("query", ""),
                        "metadata": {
                            "source": "ms_marco",
                            "query_type": item.get("query_type", ""),
                        }
                    })

                    if len(passages) >= num_passages:
                        break

            print(f"  Loaded {len(passages)} MS MARCO passages")
            return passages

        except Exception as e:
            print(f"  Error loading MS MARCO: {e}")
            print("  Trying alternative dataset...")
            return self.load_squad_contexts(num_passages)

    def load_squad_contexts(self, num_contexts: int = 500) -> List[Dict]:
        """
        Load SQuAD dataset contexts (Wikipedia paragraphs with Q&A).
        Good quality, diverse content.
        """
        print(f"Loading {num_contexts} SQuAD contexts...")

        try:
            from datasets import load_dataset

            dataset = load_dataset("squad", split="train", trust_remote_code=True)

            contexts = []
            seen = set()

            for item in tqdm(dataset, total=len(dataset), desc="Loading SQuAD"):
                if len(contexts) >= num_contexts:
                    break

                context = item.get("context", "")
                title = item.get("title", "")

                # Deduplicate
                context_hash = hash(context[:200])
                if context_hash in seen:
                    continue
                seen.add(context_hash)

                if len(context) < 200:
                    continue

                contexts.append({
                    "id": f"squad_{len(contexts):04d}",
                    "type": "document_chunk",
                    "source": "squad",
                    "title": title,
                    "content": context,
                    "metadata": {
                        "source": "squad",
                        "title": title,
                        "has_qa": True,
                        "sample_question": item.get("question", ""),
                    }
                })

            print(f"  Loaded {len(contexts)} SQuAD contexts")
            return contexts

        except Exception as e:
            print(f"  Error loading SQuAD: {e}")
            return []

    def load_paul_graham_essays(self) -> List[Dict]:
        """
        Load Paul Graham's essays - popular dataset for RAG demos.
        Clean, well-written content about startups/tech.
        """
        print("Loading Paul Graham essays...")

        essays = []

        # Paul Graham essay URLs (public domain for educational use)
        essay_titles = [
            "How to Start a Startup",
            "Do Things that Don't Scale",
            "Maker's Schedule, Manager's Schedule",
            "The Python Paradox",
            "Great Hackers",
            "What You'll Wish You'd Known",
            "How to Get Startup Ideas",
            "Startup = Growth",
            "Mean People Fail",
            "The Hardest Lessons for Startups to Learn",
        ]

        # Use a pre-compiled dataset of PG essays
        try:
            from datasets import load_dataset

            # Try loading from HuggingFace if available
            dataset = load_dataset(
                "jamescalam/paul-graham-essays",
                split="train",
                trust_remote_code=True
            )

            for i, item in enumerate(dataset):
                content = item.get("text", "")
                title = item.get("title", f"Essay {i}")

                # Chunk into ~500 word pieces
                words = content.split()
                chunk_size = 500
                for j in range(0, len(words), chunk_size):
                    chunk = " ".join(words[j:j + chunk_size])
                    if len(chunk) > 200:
                        essays.append({
                            "id": f"pg_{len(essays):04d}",
                            "type": "document_chunk",
                            "source": "paul_graham",
                            "title": title,
                            "content": chunk,
                            "metadata": {
                                "source": "paul_graham_essays",
                                "title": title,
                                "chunk_index": j // chunk_size,
                            }
                        })

            print(f"  Loaded {len(essays)} Paul Graham essay chunks")
            return essays

        except Exception as e:
            print(f"  Paul Graham dataset not available: {e}")
            return []

    def load_news_articles(self, num_articles: int = 500) -> List[Dict]:
        """
        Load AG News or BBC News dataset.
        Real news articles across categories.
        """
        print(f"Loading {num_articles} news articles...")

        try:
            from datasets import load_dataset

            # AG News: 4 categories (World, Sports, Business, Sci/Tech)
            dataset = load_dataset("ag_news", split="train", trust_remote_code=True)

            articles = []
            for item in tqdm(dataset, total=min(num_articles, len(dataset)), desc="Loading AG News"):
                if len(articles) >= num_articles:
                    break

                text = item.get("text", "")
                label = item.get("label", 0)
                categories = ["World", "Sports", "Business", "Sci/Tech"]

                if len(text) > 100:
                    articles.append({
                        "id": f"news_{len(articles):04d}",
                        "type": "text",
                        "source": "ag_news",
                        "content": text,
                        "metadata": {
                            "source": "ag_news",
                            "category": categories[label] if label < len(categories) else "Unknown",
                        }
                    })

            print(f"  Loaded {len(articles)} news articles")
            return articles

        except Exception as e:
            print(f"  Error loading news: {e}")
            return []

    def load_stackoverflow(self, num_posts: int = 300) -> List[Dict]:
        """
        Load Stack Overflow Q&A data.
        Programming questions and answers.
        """
        print(f"Loading {num_posts} Stack Overflow posts...")

        try:
            from datasets import load_dataset

            # Stack Overflow dataset
            dataset = load_dataset(
                "pacovaldez/stackoverflow-questions",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            posts = []
            for item in tqdm(dataset, total=num_posts, desc="Loading StackOverflow"):
                if len(posts) >= num_posts:
                    break

                title = item.get("title", "")
                body = item.get("body", "")
                tags = item.get("tags", "")

                content = f"Question: {title}\n\n{body[:800]}"

                if len(content) > 200:
                    posts.append({
                        "id": f"so_{len(posts):04d}",
                        "type": "document_chunk",
                        "source": "stackoverflow",
                        "title": title,
                        "content": content,
                        "metadata": {
                            "source": "stackoverflow",
                            "tags": tags,
                        }
                    })

            print(f"  Loaded {len(posts)} Stack Overflow posts")
            return posts

        except Exception as e:
            print(f"  Error loading Stack Overflow: {e}")
            return []

    def load_pubmed_abstracts(self, num_abstracts: int = 200) -> List[Dict]:
        """
        Load PubMed medical/scientific abstracts.
        Good for testing domain-specific retrieval.
        """
        print(f"Loading {num_abstracts} PubMed abstracts...")

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "pubmed_qa",
                "pqa_labeled",
                split="train",
                trust_remote_code=True
            )

            abstracts = []
            for item in tqdm(dataset, total=min(num_abstracts, len(dataset)), desc="Loading PubMed"):
                if len(abstracts) >= num_abstracts:
                    break

                context = item.get("context", {})
                contexts_list = context.get("contexts", [])
                question = item.get("question", "")

                for ctx in contexts_list:
                    if len(ctx) > 200:
                        abstracts.append({
                            "id": f"pubmed_{len(abstracts):04d}",
                            "type": "document_chunk",
                            "source": "pubmed",
                            "content": ctx,
                            "metadata": {
                                "source": "pubmed_qa",
                                "related_question": question[:100],
                            }
                        })

                        if len(abstracts) >= num_abstracts:
                            break

            print(f"  Loaded {len(abstracts)} PubMed abstracts")
            return abstracts

        except Exception as e:
            print(f"  Error loading PubMed: {e}")
            return []

    def load_financial_news(self, num_articles: int = 300) -> List[Dict]:
        """
        Load financial news and sentiment data.
        Good for finance/business use cases.
        """
        print(f"Loading {num_articles} financial articles...")

        try:
            from datasets import load_dataset

            # Financial PhraseBank - sentences with sentiment
            dataset = load_dataset(
                "financial_phrasebank",
                "sentences_50agree",
                split="train",
                trust_remote_code=True
            )

            articles = []
            for item in tqdm(dataset, total=min(num_articles, len(dataset)), desc="Loading Financial"):
                if len(articles) >= num_articles:
                    break

                sentence = item.get("sentence", "")
                label = item.get("label", 0)
                sentiments = ["negative", "neutral", "positive"]

                if len(sentence) > 50:
                    articles.append({
                        "id": f"fin_{len(articles):04d}",
                        "type": "text",
                        "source": "financial",
                        "content": sentence,
                        "metadata": {
                            "source": "financial_phrasebank",
                            "sentiment": sentiments[label] if label < len(sentiments) else "unknown",
                        }
                    })

            print(f"  Loaded {len(articles)} financial sentences")
            return articles

        except Exception as e:
            print(f"  Error loading financial data: {e}")
            return []

    def load_coco_images(self, num_images: int = 50) -> List[Dict]:
        """
        Load COCO dataset images with captions.
        Real photos with human-written descriptions.
        """
        print(f"Loading {num_images} COCO images...")

        images_dir = Path(config.IMAGES_DIR)
        images_dir.mkdir(exist_ok=True)

        try:
            from datasets import load_dataset

            # COCO captions dataset
            dataset = load_dataset(
                "HuggingFaceM4/COCO",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            images = []
            for item in tqdm(dataset, total=num_images, desc="Loading COCO"):
                if len(images) >= num_images:
                    break

                try:
                    image = item.get("image")
                    captions = item.get("sentences", {}).get("raw", [])
                    caption = captions[0] if captions else "No caption"

                    # Save image
                    filename = f"coco_{len(images):03d}.jpg"
                    filepath = images_dir / filename

                    if image is not None:
                        image.save(filepath)

                        images.append({
                            "id": f"img_{len(images):04d}",
                            "type": "image",
                            "filepath": str(filepath),
                            "source": "coco",
                            "description": caption,
                            "metadata": {
                                "source": "coco",
                                "caption": caption,
                            }
                        })
                except Exception as e:
                    continue

            print(f"  Loaded {len(images)} COCO images")
            return images

        except Exception as e:
            print(f"  Error loading COCO: {e}")
            return []

    def load_github_readmes(self, num_repos: int = 100) -> List[Dict]:
        """
        Load GitHub repository READMEs.
        Real technical documentation.
        """
        print(f"Loading {num_repos} GitHub READMEs...")

        try:
            from datasets import load_dataset

            # GitHub Code dataset subset
            dataset = load_dataset(
                "codeparrot/github-code",
                streaming=True,
                split="train",
                languages=["Markdown"],
                trust_remote_code=True
            )

            readmes = []
            for item in tqdm(dataset, total=num_repos * 10, desc="Loading GitHub"):
                if len(readmes) >= num_repos:
                    break

                path = item.get("path", "").lower()
                code = item.get("code", "")

                # Only README files
                if "readme" not in path:
                    continue

                if len(code) > 500:
                    # Take first 2000 chars
                    content = code[:2000]

                    readmes.append({
                        "id": f"gh_{len(readmes):04d}",
                        "type": "document_chunk",
                        "source": "github",
                        "content": content,
                        "metadata": {
                            "source": "github_readme",
                            "repo": item.get("repo_name", ""),
                            "path": path,
                        }
                    })

            print(f"  Loaded {len(readmes)} GitHub READMEs")
            return readmes

        except Exception as e:
            print(f"  Error loading GitHub: {e}")
            return []

    def load_tech_images(self, num_images: int = 50) -> List[Dict]:
        """
        Download real tech/diagram images from public sources.
        Uses Unsplash API (free) or Wikimedia Commons.
        """
        print(f"Loading {num_images} real tech images...")

        images_dir = Path(config.IMAGES_DIR)
        images_dir.mkdir(exist_ok=True)

        images = []

        # Option 1: Use picsum.photos for placeholder images
        # Option 2: Download from Wikimedia Commons

        # Wikimedia Commons categories for tech diagrams
        wikimedia_files = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Kubernetes_logo.svg/200px-Kubernetes_logo.svg.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Docker_%28container_engine%29_logo.svg/200px-Docker_%28container_engine%29_logo.svg.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Amazon_Web_Services_Logo.svg/200px-Amazon_Web_Services_Logo.svg.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Postgresql_elephant.svg/200px-Postgresql_elephant.svg.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Expressjs.png/200px-Expressjs.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/200px-React-icon.svg.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/200px-Octicons-mark-github.svg.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Jenkins_logo.svg/200px-Jenkins_logo.svg.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Ansible_logo.svg/200px-Ansible_logo.svg.png",
        ]

        # Download images
        for i, url in enumerate(tqdm(wikimedia_files[:num_images], desc="Downloading images")):
            try:
                filename = f"tech_image_{i:03d}.png"
                filepath = images_dir / filename

                if not filepath.exists():
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(filepath, "wb") as f:
                            f.write(response.content)

                if filepath.exists():
                    # Extract name from URL
                    name = url.split("/")[-1].replace("200px-", "").replace(".png", "").replace(".svg", "")

                    images.append({
                        "id": f"img_{i:04d}",
                        "type": "image",
                        "filepath": str(filepath),
                        "source": "wikimedia",
                        "description": f"Technology logo/diagram: {name}",
                        "metadata": {
                            "source": "wikimedia_commons",
                            "name": name,
                            "url": url,
                        }
                    })
            except Exception as e:
                print(f"  Error downloading {url}: {e}")
                continue

        # If we need more images, generate some with descriptions
        if len(images) < num_images:
            print(f"  Generating {num_images - len(images)} additional diagram images...")
            from data_generator import generate_images
            generated = generate_images(num_images - len(images))

            # Update IDs
            for j, img in enumerate(generated):
                img["id"] = f"img_{len(images) + j:04d}"
                images.append(img)

        print(f"  Loaded {len(images)} images")
        return images

    def load_all(
        self,
        num_texts: int = 1000,
        num_chunks: int = 200,
        num_images: int = 50,
        sources: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Load data from multiple public sources.

        Args:
            sources: List of sources to use. Options:
                     Text: ["wikipedia", "news", "financial"]
                     Chunks: ["arxiv", "msmarco", "squad", "paul_graham",
                              "stackoverflow", "pubmed", "github_readme"]
                     Images: ["coco", "diagrams"]
                     Default: ["wikipedia", "squad"]
        """
        if sources is None:
            sources = ["wikipedia", "squad"]

        print("=" * 60)
        print("Loading Public Datasets for RAG Benchmark")
        print("=" * 60)
        print(f"Sources: {', '.join(sources)}")
        print("=" * 60)

        text_documents = []
        document_chunks = []

        # Load text documents (shorter content)
        remaining_texts = num_texts

        if "wikipedia" in sources and remaining_texts > 0:
            wiki_docs = self.load_wikipedia_articles(remaining_texts)
            text_documents.extend(wiki_docs)
            remaining_texts -= len(wiki_docs)

        if "news" in sources and remaining_texts > 0:
            news_docs = self.load_news_articles(remaining_texts)
            text_documents.extend(news_docs)
            remaining_texts -= len(news_docs)

        if "financial" in sources and remaining_texts > 0:
            fin_docs = self.load_financial_news(remaining_texts)
            text_documents.extend(fin_docs)
            remaining_texts -= len(fin_docs)

        # Load document chunks (longer content)
        remaining_chunks = num_chunks

        if "arxiv" in sources and remaining_chunks > 0:
            arxiv_docs = self.load_arxiv_abstracts(min(remaining_chunks, 100))
            document_chunks.extend(arxiv_docs)
            remaining_chunks -= len(arxiv_docs)

        if "squad" in sources and remaining_chunks > 0:
            squad_docs = self.load_squad_contexts(min(remaining_chunks, 200))
            document_chunks.extend(squad_docs)
            remaining_chunks -= len(squad_docs)

        if "msmarco" in sources and remaining_chunks > 0:
            msmarco_docs = self.load_msmarco_passages(min(remaining_chunks, 200))
            document_chunks.extend(msmarco_docs)
            remaining_chunks -= len(msmarco_docs)

        if "stackoverflow" in sources and remaining_chunks > 0:
            so_docs = self.load_stackoverflow(min(remaining_chunks, 150))
            document_chunks.extend(so_docs)
            remaining_chunks -= len(so_docs)

        if "pubmed" in sources and remaining_chunks > 0:
            pubmed_docs = self.load_pubmed_abstracts(min(remaining_chunks, 100))
            document_chunks.extend(pubmed_docs)
            remaining_chunks -= len(pubmed_docs)

        if "github_readme" in sources and remaining_chunks > 0:
            gh_docs = self.load_github_readmes(min(remaining_chunks, 100))
            document_chunks.extend(gh_docs)
            remaining_chunks -= len(gh_docs)

        if "paul_graham" in sources:
            pg_docs = self.load_paul_graham_essays()
            document_chunks.extend(pg_docs[:100])  # Limit

        # Load images
        images = []
        if "coco" in sources:
            coco_imgs = self.load_coco_images(num_images)
            images.extend(coco_imgs)

        # Fallback to tech images/diagrams
        if len(images) < num_images:
            tech_imgs = self.load_tech_images(num_images - len(images))
            images.extend(tech_imgs)

        # Ensure we have enough data (pad with what we have if needed)
        if len(text_documents) < num_texts:
            print(f"  Note: Only loaded {len(text_documents)} text documents (requested {num_texts})")

        if len(document_chunks) < num_chunks:
            print(f"  Note: Only loaded {len(document_chunks)} document chunks (requested {num_chunks})")

        data = {
            "text_documents": text_documents[:num_texts],
            "document_chunks": document_chunks[:num_chunks],
            "images": images[:num_images]
        }

        print("\n" + "=" * 60)
        print("Data Loading Complete!")
        print("=" * 60)
        print(f"  Text documents: {len(data['text_documents'])} (from {', '.join(sources)})")
        print(f"  Document chunks: {len(data['document_chunks'])}")
        print(f"  Images: {len(data['images'])}")

        return data


def get_sample_queries_for_dataset(data: Dict) -> List[Dict]:
    """Generate relevant queries based on loaded data."""
    queries = []

    # Extract topics/titles from data for relevant queries
    for doc in data.get("text_documents", [])[:20]:
        title = doc.get("title", doc.get("metadata", {}).get("title", ""))
        if title:
            queries.append({
                "query": f"What is {title}?",
                "type": "text",
                "source": doc.get("source", "unknown")
            })
            queries.append({
                "query": f"Explain {title}",
                "type": "text",
                "source": doc.get("source", "unknown")
            })

    # Add some generic tech queries
    generic_queries = [
        "How does machine learning work?",
        "What is cloud computing?",
        "Explain neural networks",
        "What is Docker used for?",
        "How do databases work?",
        "What is API design?",
        "Explain microservices architecture",
        "What is DevOps?",
        "How does encryption work?",
        "What is Kubernetes?",
    ]

    for q in generic_queries:
        queries.append({"query": q, "type": "text", "source": "generic"})

    return queries


if __name__ == "__main__":
    loader = PublicDataLoader()

    # Test loading different sources
    print("\nAvailable datasets:")
    for name, desc in DATASETS.items():
        print(f"  {name}: {desc}")

    # Load a small sample
    data = loader.load_all(
        num_texts=100,
        num_chunks=50,
        num_images=10,
        sources=["wikipedia", "squad"]
    )

    # Show samples
    print("\n\nSample text document:")
    if data["text_documents"]:
        doc = data["text_documents"][0]
        print(f"  Source: {doc.get('source')}")
        print(f"  Title: {doc.get('title', 'N/A')}")
        print(f"  Content: {doc.get('content', '')[:200]}...")

    print("\n\nSample document chunk:")
    if data["document_chunks"]:
        chunk = data["document_chunks"][0]
        print(f"  Source: {chunk.get('source')}")
        print(f"  Content: {chunk.get('content', '')[:200]}...")
