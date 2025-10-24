"""Tests for the LangGraph RAG agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.docarag.services.agent import (
    AgentState,
    rephrase_query_node,
    embed_query_node,
    retrieve_documents_node,
    generate_answer_node,
    evaluate_answer_node,
    should_continue,
)


@pytest.mark.asyncio
async def test_rephrase_query_node():
    """Test query rephrasing node."""
    state = AgentState(
        query="What is Python?",
        max_iterations=2,
    )
    
    with patch("src.docarag.services.agent.ChatAnthropic") as mock_llm:
        mock_instance = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "What are the key features of Python programming language?"
        mock_instance.ainvoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        
        result = await rephrase_query_node(state)
        
        assert result["rephrased_query"] is not None
        assert len(result["rephrased_query"]) > 0


@pytest.mark.asyncio
async def test_embed_query_node():
    """Test query embedding node."""
    state = AgentState(
        query="What is Python?",
        rephrased_query="What are the key features of Python?",
        max_iterations=2,
    )
    
    with patch("src.docarag.services.agent.EmbeddingGRPCClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.embed_text_async.return_value = [0.1] * 768
        mock_client.return_value = mock_instance
        
        result = await embed_query_node(state)
        
        assert result["query_embedding"] is not None
        assert len(result["query_embedding"]) == 768


@pytest.mark.asyncio
async def test_retrieve_documents_node():
    """Test document retrieval node."""
    state = AgentState(
        query="What is Python?",
        rephrased_query="What are the key features of Python?",
        query_embedding=[0.1] * 768,
        max_iterations=2,
    )
    
    mock_obj = MagicMock()
    mock_obj.properties = {
        "content": "Python is a high-level programming language.",
        "document_name": "test.pdf",
        "page": 1,
        "date_created": "2024-01-01",
    }
    mock_obj.metadata.distance = 0.2
    
    mock_response = MagicMock()
    mock_response.objects = [mock_obj]
    
    with patch("src.docarag.services.agent.get_vector_db_client") as mock_client:
        mock_collection = AsyncMock()
        mock_query = AsyncMock()
        mock_query.do.return_value = mock_response
        mock_collection.query.near_vector.return_value = mock_query
        
        mock_client_instance = AsyncMock()
        mock_client_instance.collections.get.return_value = mock_collection
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        result = await retrieve_documents_node(state)
        
        assert len(result["retrieved_docs"]) > 0
        assert result["retrieved_docs"][0]["content"] == "Python is a high-level programming language."


@pytest.mark.asyncio
async def test_generate_answer_node():
    """Test answer generation node."""
    state = AgentState(
        query="What is Python?",
        rephrased_query="What are the key features of Python?",
        query_embedding=[0.1] * 768,
        retrieved_docs=[
            {
                "content": "Python is a high-level programming language.",
                "document_name": "test.pdf",
                "page": 1,
                "similarity_score": 0.9,
            }
        ],
        max_iterations=2,
    )
    
    with patch("src.docarag.services.agent.ChatAnthropic") as mock_llm:
        mock_instance = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Python is a high-level programming language known for its simplicity."
        mock_instance.ainvoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        
        result = await generate_answer_node(state)
        
        assert result["answer"] is not None
        assert len(result["answer"]) > 0


@pytest.mark.asyncio
async def test_evaluate_answer_node():
    """Test answer evaluation node."""
    state = AgentState(
        query="What is Python?",
        rephrased_query="What are the key features of Python?",
        query_embedding=[0.1] * 768,
        retrieved_docs=[{"content": "Python is a programming language."}],
        answer="Python is a high-level programming language.",
        max_iterations=2,
    )
    
    with patch("src.docarag.services.agent.ChatAnthropic") as mock_llm:
        mock_instance = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "0.85"
        mock_instance.ainvoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        
        result = await evaluate_answer_node(state)
        
        assert result["confidence"] > 0
        assert result["confidence"] <= 1.0


def test_should_continue():
    """Test conditional routing logic."""
    state_continue = AgentState(
        query="test",
        confidence=0.5,
        iterations=1,
        should_iterate=True,
        max_iterations=2,
    )
    
    assert should_continue(state_continue) == "rephrase_query"
    
    state_end = AgentState(
        query="test",
        confidence=0.9,
        iterations=1,
        should_iterate=False,
        max_iterations=2,
    )
    
    assert should_continue(state_end) == "end"

