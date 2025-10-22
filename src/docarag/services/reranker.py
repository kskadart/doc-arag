# from typing import List, Dict, Any, Optional
# from sentence_transformers import CrossEncoder
# from src.docarag.config import settings


# class RerankerService:
#     """Service for reranking documents using a cross-encoder model."""

#     def __init__(self, model_name: Optional[str] = None):
#         """
#         Initialize reranker service.

#         Args:
#             model_name: Name of the cross-encoder model
#         """
#         self.model_name = model_name or settings.reranker_model_name
#         self.model: Optional[CrossEncoder] = None

#     def load_model(self) -> None:
#         """Load the reranker model."""
#         if self.model is None:
#             self.model = CrossEncoder(self.model_name)

#     def rerank(
#         self,
#         query: str,
#         documents: List[Dict[str, Any]],
#         top_k: Optional[int] = None,
#         score_key: str = "content"
#     ) -> List[Dict[str, Any]]:
#         """
#         Rerank documents based on relevance to query.

#         Args:
#             query: Search query
#             documents: List of document dictionaries
#             top_k: Number of top results to return
#             score_key: Key in document dict containing text to score

#         Returns:
#             Reranked and scored documents

#         Raises:
#             ValueError: If query or documents are empty
#             Exception: If reranking fails
#         """
#         if not query or not query.strip():
#             raise ValueError("Query cannot be empty")

#         if not documents:
#             return []

#         if top_k is None:
#             top_k = settings.rerank_top_k

#         if self.model is None:
#             self.load_model()

#         try:
#             # Prepare query-document pairs for scoring
#             pairs = [[query, doc.get(score_key, "")] for doc in documents]

#             # Get scores from cross-encoder
#             scores = self.model.predict(pairs)

#             # Attach scores to documents
#             scored_docs = []
#             for doc, score in zip(documents, scores):
#                 doc_copy = doc.copy()
#                 doc_copy["rerank_score"] = float(score)
#                 scored_docs.append(doc_copy)

#             # Sort by score in descending order
#             scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

#             # Return top-k results
#             return scored_docs[:top_k]

#         except Exception as e:
#             raise Exception(f"Failed to rerank documents: {str(e)}")

#     def score_pairs(self, query: str, texts: List[str]) -> List[float]:
#         """
#         Score query-text pairs.

#         Args:
#             query: Search query
#             texts: List of texts to score

#         Returns:
#             List of relevance scores

#         Raises:
#             ValueError: If inputs are invalid
#             Exception: If scoring fails
#         """
#         if not query or not query.strip():
#             raise ValueError("Query cannot be empty")

#         if not texts:
#             return []

#         if self.model is None:
#             self.load_model()

#         try:
#             pairs = [[query, text] for text in texts]
#             scores = self.model.predict(pairs)
#             return [float(s) for s in scores]

#         except Exception as e:
#             raise Exception(f"Failed to score pairs: {str(e)}")


# # Global reranker service instance
# reranker_service: Optional[RerankerService] = None


# def get_reranker_service() -> RerankerService:
#     """Get or create reranker service instance."""
#     global reranker_service
#     if reranker_service is None:
#         reranker_service = RerankerService()
#     return reranker_service
