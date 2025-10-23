"""Background tasks for document processing."""

from src.docarag.tasks.embedding_task import run_embedding_task

__all__ = ["run_embedding_task"]
