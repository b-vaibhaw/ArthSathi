"""
vector_store.py
================
Semantic search over the financial knowledge base. Uses a multilingual
sentence-transformers model + FAISS when those optional packages are
installed; otherwise falls back to a dependency-free keyword overlap search
so the platform still runs (just with lower-quality retrieval) without
those extra downloads.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional


class VectorStore:
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    DIM = 384

    def __init__(self, index_path: Optional[str] = None, use_embeddings: bool = True):
        self.encoder = None
        self.index = None
        self.documents: List[Dict] = []
        if use_embeddings:
            self._load_encoder()
        if index_path and Path(index_path).exists():
            self.load(index_path)

    def _load_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.MODEL_NAME)
        except ImportError:
            pass  # keyword fallback used instead

    def _embed(self, text: str):
        import numpy as np
        if self.encoder:
            v = self.encoder.encode([text])[0].astype("float32")
        else:
            rng = __import__("random").Random(int(hashlib.md5(text.encode()).hexdigest(), 16))
            v = np.array([rng.gauss(0, 1) for _ in range(self.DIM)], dtype="float32")
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)

    def _ensure_index(self):
        if self.index is None:
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.DIM)
            except ImportError:
                pass

    def add_batch(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        self._ensure_index()
        metadatas = metadatas or [{} for _ in texts]
        if self.index is not None and self.encoder is not None:
            import numpy as np
            embs = np.stack([self._embed(t) for t in texts])
            self.index.add(embs)
        for t, m in zip(texts, metadatas):
            self.documents.append({"text": t, "metadata": m or {}})

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not self.documents:
            return []
        if self.index is None or self.encoder is None or getattr(self.index, "ntotal", 0) == 0:
            return self._keyword_search(query, k)
        q = self._embed(query).reshape(1, -1)
        scores, idxs = self.index.search(q, min(k, len(self.documents)))
        results = []
        for sc, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(self.documents):
                d = dict(self.documents[i])
                d["score"] = float(sc)
                results.append(d)
        return results

    def _keyword_search(self, query: str, k: int) -> List[Dict]:
        qw = set(query.lower().split())
        scored = sorted(
            [(len(qw & set(d["text"].lower().split())), d) for d in self.documents],
            key=lambda x: x[0], reverse=True,
        )
        return [d for sc, d in scored[:k] if sc > 0]

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            try:
                import faiss
                faiss.write_index(self.index, str(Path(path) / "faiss.index"))
            except ImportError:
                pass
        with open(Path(path) / "docs.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def load(self, path: str):
        try:
            import faiss
            idx_file = Path(path) / "faiss.index"
            if idx_file.exists():
                self.index = faiss.read_index(str(idx_file))
        except ImportError:
            pass
        docs_file = Path(path) / "docs.json"
        if docs_file.exists():
            with open(docs_file, encoding="utf-8") as f:
                self.documents = json.load(f)
