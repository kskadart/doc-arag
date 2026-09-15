import os
import sys
from unittest.mock import Mock

os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LLM_BASE_URL", "http://llm.test/v1")
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("LLM_MODEL", "test/chat-model")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://embed.test/v1")
os.environ.setdefault("EMBEDDING_API_KEY", "test-embed-key")
os.environ.setdefault("EMBEDDING_MODEL", "test/embedding-model")
os.environ.setdefault("RERANKER_PROVIDER", "grpc")
os.environ.setdefault("RERANKER_SERVICE_URL", "localhost:8352")
os.environ.setdefault("STARTUP_VERIFY_EMBEDDING_DIMENSION", "false")
# Login disabled under test regardless of the developer's .env
os.environ.setdefault("AUTH_TRUSTED_HEADERS", "false")
os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "test-access-key")
os.environ.setdefault("MINIO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("MINIO_BUCKET", "test-bucket")
os.environ.setdefault("MINIO_SECURE", "false")
os.environ.setdefault("WEAVIATE_HOST", "localhost")
os.environ.setdefault("WEAVIATE_PORT", "8080")

mock_rag_agent_module = Mock()
mock_rag_agent_module.get_rag_agent = Mock()
mock_rag_agent_module.RAGAgent = Mock()
mock_rag_agent_module.RAGState = Mock()
sys.modules["src.docarag.services.rag_agent"] = mock_rag_agent_module

mock_background_tasks_module = Mock()
mock_background_tasks_module.process_scraping_task = Mock()
mock_background_tasks_module.process_embedding_task = Mock()
mock_background_tasks_module.create_task_id = Mock(return_value="test-task-id")
mock_background_tasks_module.get_task_status = Mock(return_value={"status": "pending"})
sys.modules["src.docarag.utils.background_tasks"] = mock_background_tasks_module
