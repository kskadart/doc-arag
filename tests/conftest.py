import os
import sys
from unittest.mock import Mock

os.environ.setdefault("ANTHROPIC_API_KEY", "test-api-key-123")
os.environ.setdefault("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "test-access-key")
os.environ.setdefault("MINIO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("MINIO_BUCKET", "test-bucket")
os.environ.setdefault("MINIO_SECURE", "false")
os.environ.setdefault("WEAVIATE_HOST", "localhost")
os.environ.setdefault("WEAVIATE_PORT", "8080")
os.environ.setdefault("WEAVIATE_COLLECTION", "TestDocuments")
os.environ.setdefault("EMBEDDING_SERVICE_URL", "localhost:8351")

mock_rag_agent_module = Mock()
mock_rag_agent_module.get_rag_agent = Mock()
mock_rag_agent_module.RAGAgent = Mock()
mock_rag_agent_module.RAGState = Mock()
sys.modules['src.docarag.services.rag_agent'] = mock_rag_agent_module

mock_background_tasks_module = Mock()
mock_background_tasks_module.process_scraping_task = Mock()
mock_background_tasks_module.process_embedding_task = Mock()
mock_background_tasks_module.create_task_id = Mock(return_value="test-task-id")
mock_background_tasks_module.get_task_status = Mock(return_value={"status": "pending"})
sys.modules['src.docarag.utils.background_tasks'] = mock_background_tasks_module
