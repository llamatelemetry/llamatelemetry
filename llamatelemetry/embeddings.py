"""
llamatelemetry.embeddings - Text Embedding Support

This module provides text embedding generation with:
- Single and batch embedding generation
- OpenAI-compatible embedding API
- Similarity search utilities
- Embedding caching for efficiency
- Support for different pooling strategies

Examples:
    Basic embedding:
    >>> from llamatelemetry.embeddings import EmbeddingEngine
    >>> embedder = EmbeddingEngine(engine)
    >>> vector = embedder.embed("Hello world")
    >>> print(vector.shape)  # (768,) or model-specific dimension

    Batch embedding:
    >>> texts = ["First text", "Second text", "Third text"]
    >>> vectors = embedder.embed_batch(texts)
    >>> print(vectors.shape)  # (3, 768)

    Similarity search:
    >>> from llamatelemetry.embeddings import cosine_similarity
    >>> similarity = cosine_similarity(vector1, vector2)
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import requests
import time
from pathlib import Path
import json


class EmbeddingEngine:
    """
    Text embedding generation engine.

    Provides methods for generating embeddings from text using llama-server's
    embedding endpoint with support for caching and batch processing.

    Examples:
        >>> from llamatelemetry import InferenceEngine
        >>> from llamatelemetry.embeddings import EmbeddingEngine
        >>> engine = InferenceEngine()
        >>> engine.load_model("model.gguf", auto_start=True)
        >>> embedder = EmbeddingEngine(engine)
        >>> embedding = embedder.embed("AI is transforming the world")
        >>> print(f"Embedding dimension: {len(embedding)}")
    """

    def __init__(
        self,
        engine,
        pooling: str = "mean",
        normalize: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize embedding engine.

        Args:
            engine: InferenceEngine instance
            pooling: Pooling strategy (mean, cls, last)
            normalize: Normalize embeddings to unit vectors
            cache_size: Maximum number of embeddings to cache
        """
        self.engine = engine
        self.pooling = pooling
        self.normalize = normalize
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            use_cache: Use cached embedding if available

        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        if use_cache and text in self.cache:
            self.cache_hits += 1
            return self.cache[text]

        self.cache_misses += 1

        # Try OpenAI-compatible endpoint first
        try:
            response = requests.post(
                f"{self.engine.server_url}/v1/embeddings",
                json={
                    "input": text,
                    "encoding_format": "float"
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)

                if self.normalize:
                    embedding = embedding / np.linalg.norm(embedding)

                # Cache result
                if use_cache:
                    self._add_to_cache(text, embedding)

                return embedding

        except Exception:
            pass

        # Fallback to native endpoint
        try:
            response = requests.post(
                f"{self.engine.server_url}/embedding",
                json={
                    "content": text,
                    "pooling": self.pooling
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                embedding = np.array(data["embedding"], dtype=np.float32)

                if self.normalize:
                    embedding = embedding / np.linalg.norm(embedding)

                # Cache result
                if use_cache:
                    self._add_to_cache(text, embedding)

                return embedding

        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")

        raise RuntimeError("Embedding endpoint not available")

    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            use_cache: Use cached embeddings if available
            show_progress: Show progress bar

        Returns:
            2D numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = []

        if show_progress:
            try:
                from tqdm.auto import tqdm
                texts_iter = tqdm(texts, desc="Embedding")
            except ImportError:
                texts_iter = texts
        else:
            texts_iter = texts

        for text in texts_iter:
            embedding = self.embed(text, use_cache=use_cache)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def _add_to_cache(self, text: str, embedding: np.ndarray):
        """Add embedding to cache with LRU eviction."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO, could be improved to LRU)
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        self.cache[text] = embedding

    def clear_cache(self):
        """Clear embedding cache."""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_size": len(self.cache),
            "cache_max": self.cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate
        }

    def save_cache(self, filepath: str):
        """
        Save cache to disk.

        Args:
            filepath: Path to save cache
        """
        data = {
            "embeddings": {
                text: embedding.tolist()
                for text, embedding in self.cache.items()
            },
            "metadata": {
                "pooling": self.pooling,
                "normalize": self.normalize,
                "timestamp": time.time()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load_cache(self, filepath: str):
        """
        Load cache from disk.

        Args:
            filepath: Path to cache file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.cache = {
            text: np.array(embedding, dtype=np.float32)
            for text, embedding in data["embeddings"].items()
        }


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0 to 1 for normalized vectors)
    """
    # Normalize if not already
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

    return float(np.dot(vec1_norm, vec2_norm))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(vec1 - vec2))


def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute dot product similarity (for normalized vectors).

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Dot product similarity
    """
    return float(np.dot(vec1, vec2))


class SemanticSearch:
    """
    Simple semantic search using embeddings.

    Allows indexing documents and searching for similar ones
    using vector similarity.

    Examples:
        >>> search = SemanticSearch(embedder)
        >>> search.add_documents([
        ...     "Python is a programming language",
        ...     "Machine learning is a subset of AI",
        ...     "Natural language processing uses deep learning"
        ... ])
        >>> results = search.search("What is NLP?", top_k=2)
        >>> for doc, score in results:
        ...     print(f"{score:.3f}: {doc}")
    """

    def __init__(self, embedder: EmbeddingEngine):
        """
        Initialize semantic search.

        Args:
            embedder: EmbeddingEngine instance
        """
        self.embedder = embedder
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = False
    ):
        """
        Add documents to search index.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            show_progress: Show progress bar
        """
        # Generate embeddings
        new_embeddings = self.embedder.embed_batch(documents, show_progress=show_progress)

        # Add to index
        self.documents.extend(documents)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Add metadata
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_fn: str = "cosine"
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return
            similarity_fn: Similarity function (cosine, dot, euclidean)

        Returns:
            List of (document, score, metadata) tuples
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Embed query
        query_embedding = self.embedder.embed(query)

        # Compute similarities
        if similarity_fn == "cosine":
            scores = np.array([
                cosine_similarity(query_embedding, doc_emb)
                for doc_emb in self.embeddings
            ])
        elif similarity_fn == "dot":
            scores = np.array([
                dot_product_similarity(query_embedding, doc_emb)
                for doc_emb in self.embeddings
            ])
        elif similarity_fn == "euclidean":
            scores = -np.array([
                euclidean_distance(query_embedding, doc_emb)
                for doc_emb in self.embeddings
            ])
        else:
            raise ValueError(f"Unknown similarity function: {similarity_fn}")

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return results
        results = [
            (self.documents[idx], float(scores[idx]), self.metadata[idx])
            for idx in top_indices
        ]

        return results

    def save_index(self, filepath: str):
        """
        Save search index to disk.

        Args:
            filepath: Path to save index
        """
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "metadata": self.metadata
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load_index(self, filepath: str):
        """
        Load search index from disk.

        Args:
            filepath: Path to index file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.documents = data["documents"]
        self.embeddings = np.array(data["embeddings"], dtype=np.float32) if data["embeddings"] else None
        self.metadata = data["metadata"]

    def clear_index(self):
        """Clear the search index."""
        self.documents = []
        self.embeddings = None
        self.metadata = []

    def __len__(self) -> int:
        """Return number of indexed documents."""
        return len(self.documents)


class TextClustering:
    """
    Simple text clustering using embeddings and K-means.

    Examples:
        >>> clustering = TextClustering(embedder, n_clusters=3)
        >>> texts = ["text1", "text2", "text3", ...]
        >>> labels = clustering.fit(texts)
        >>> clusters = clustering.get_clusters(texts, labels)
    """

    def __init__(self, embedder: EmbeddingEngine, n_clusters: int = 5):
        """
        Initialize text clustering.

        Args:
            embedder: EmbeddingEngine instance
            n_clusters: Number of clusters
        """
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.cluster_centers: Optional[np.ndarray] = None

    def fit(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Fit clustering model on texts.

        Args:
            texts: List of texts to cluster
            show_progress: Show progress bar

        Returns:
            Cluster labels for each text
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn required for clustering. Install with: pip install scikit-learn")

        # Generate embeddings
        embeddings = self.embedder.embed_batch(texts, show_progress=show_progress)

        # Fit K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        self.cluster_centers = kmeans.cluster_centers_

        return labels

    def get_clusters(
        self,
        texts: List[str],
        labels: np.ndarray
    ) -> Dict[int, List[str]]:
        """
        Group texts by cluster labels.

        Args:
            texts: List of texts
            labels: Cluster labels

        Returns:
            Dictionary mapping cluster ID to list of texts
        """
        clusters = {}

        for text, label in zip(texts, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text)

        return clusters

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict cluster labels for new texts.

        Args:
            texts: List of texts

        Returns:
            Predicted cluster labels
        """
        if self.cluster_centers is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        embeddings = self.embedder.embed_batch(texts)

        # Find nearest cluster center for each embedding
        labels = []
        for embedding in embeddings:
            distances = [
                euclidean_distance(embedding, center)
                for center in self.cluster_centers
            ]
            labels.append(np.argmin(distances))

        return np.array(labels)
