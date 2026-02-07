"""\
SmartShelf AI - Semantic Search

Semantic product search using Sentence-Transformers embeddings.
"""

import logging
from typing import Dict, Any, List, Optional

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    util = None

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Semantic search for products and documents."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Semantic search model loaded: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer model ({model_name}): {e}")

        # In a real app this comes from DB/catalog. Keep small demo catalog.
        self.product_catalog = [
            {
                "product_id": "PRD-001",
                "product_name": "Organic Milk 1L",
                "category": "dairy",
                "description": "Fresh organic whole milk, 1 liter carton",
                "price": 4.99,
                "margin": 0.32,
                "in_stock": True,
            },
            {
                "product_id": "PRD-015",
                "product_name": "Organic Eggs 12ct",
                "category": "dairy",
                "description": "Free-range organic eggs, dozen pack",
                "price": 5.99,
                "margin": 0.28,
                "in_stock": True,
            },
            {
                "product_id": "PRD-023",
                "product_name": "Artisan Sourdough Bread",
                "category": "bakery",
                "description": "Handcrafted sourdough loaf, crusty and fresh",
                "price": 6.49,
                "margin": 0.40,
                "in_stock": True,
            },
            {
                "product_id": "PRD-008",
                "product_name": "Organic Coffee Beans 1lb",
                "category": "beverages",
                "description": "Single-origin organic coffee beans, medium roast",
                "price": 12.99,
                "margin": 0.48,
                "in_stock": True,
            },
        ]

        self._catalog_texts = [self._product_to_text(p) for p in self.product_catalog]
        self._catalog_embeddings = None
        if self.model:
            try:
                self._catalog_embeddings = self.model.encode(self._catalog_texts, convert_to_tensor=True)
            except Exception as e:
                logger.warning(f"Failed to embed product catalog: {e}")

    def semantic_product_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Return top-k semantic matches in product catalog."""
        filters = filters or {}

        if not query:
            return {"query": query, "products": [], "similarity_scores": []}

        # Fallback if embeddings unavailable: basic keyword match scoring
        if not self.model or self._catalog_embeddings is None or util is None:
            scored = []
            q = query.lower()
            for p in self.product_catalog:
                text = self._product_to_text(p).lower()
                score = 0.0
                for token in q.split():
                    if token in text:
                        score += 1.0
                scored.append((p, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            results = [self._apply_filters(p, filters) for p, _ in scored]
            results = [r for r in results if r is not None][:top_k]
            return {
                "query": query,
                "products": results,
                "similarity_scores": [round(float(s), 3) for _, s in scored[: len(results)]],
            }

        q_emb = self.model.encode(query, convert_to_tensor=True)
        cos = util.cos_sim(q_emb, self._catalog_embeddings)[0]
        top = cos.topk(k=min(top_k, len(self.product_catalog)))

        products: List[Dict[str, Any]] = []
        scores: List[float] = []

        for score, idx in zip(top.values, top.indices):
            p = self.product_catalog[int(idx)]
            p2 = self._apply_filters(p, filters)
            if p2 is None:
                continue
            products.append(p2)
            scores.append(float(score))

        return {
            "query": query,
            "products": products,
            "similarity_scores": [round(s, 3) for s in scores],
        }

    def _apply_filters(self, product: Dict[str, Any], filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not filters:
            return product

        if "in_stock" in filters and bool(product.get("in_stock")) != bool(filters["in_stock"]):
            return None

        if "category" in filters and product.get("category") != filters["category"]:
            return None

        if "max_price" in filters:
            try:
                if float(product.get("price", 0)) > float(filters["max_price"]):
                    return None
            except Exception:
                pass

        return product

    def _product_to_text(self, p: Dict[str, Any]) -> str:
        return f"{p.get('product_name','')} {p.get('category','')} {p.get('description','')}"
