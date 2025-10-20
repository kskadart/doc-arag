import pytest
from unittest.mock import Mock, patch
from src.docarag.services.rag_agent import RAGAgent, RAGState


@pytest.fixture
def mock_services():
    """Mock all the services needed by RAG agent."""
    with patch("src.docarag.services.rag_agent.get_embedding_service") as mock_emb, \
         patch("src.docarag.services.rag_agent.get_vectorstore_service") as mock_vs, \
         patch("src.docarag.services.rag_agent.get_reranker_service") as mock_rerank, \
         patch("src.docarag.services.rag_agent.ChatAnthropic") as mock_llm:
        
        # Setup mock returns
        mock_emb.return_value.embed_text.return_value = [0.1] * 384
        mock_vs.return_value.search.return_value = []
        mock_rerank.return_value.rerank.return_value = []
        mock_llm.return_value.invoke.return_value = Mock(content="Test response")
        
        yield {
            "embedding": mock_emb,
            "vectorstore": mock_vs,
            "reranker": mock_rerank,
            "llm": mock_llm,
        }


def test_rag_state_structure():
    """Test that RAGState has expected structure."""
    state: RAGState = {
        "query": "test",
        "rephrased_query": None,
        "file_id": None,
        "source_type": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "confidence": 0.0,
        "iterations": 0,
        "should_retry": False,
        "max_iterations": 2,
    }
    
    assert state["query"] == "test"
    assert state["iterations"] == 0


def test_rag_agent_initialization(mock_services):
    """Test RAG agent initialization."""
    agent = RAGAgent()
    assert agent is not None
    assert agent.graph is not None


def test_understand_query_node(mock_services):
    """Test query understanding node."""
    agent = RAGAgent()
    
    state: RAGState = {
        "query": "What is Python?",
        "rephrased_query": None,
        "file_id": None,
        "source_type": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "confidence": 0.0,
        "iterations": 0,
        "should_retry": False,
        "max_iterations": 2,
    }
    
    result = agent._understand_query(state)
    assert result["rephrased_query"] is not None


def test_retrieve_node(mock_services):
    """Test retrieve node."""
    agent = RAGAgent()
    
    state: RAGState = {
        "query": "test",
        "rephrased_query": "test query",
        "file_id": None,
        "source_type": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "confidence": 0.0,
        "iterations": 0,
        "should_retry": False,
        "max_iterations": 2,
    }
    
    result = agent._retrieve(state)
    assert "retrieved_docs" in result


def test_evaluate_node_low_confidence(mock_services):
    """Test evaluate node with low confidence."""
    agent = RAGAgent()
    
    state: RAGState = {
        "query": "test",
        "rephrased_query": "test",
        "file_id": None,
        "source_type": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "Short",
        "confidence": 0.0,
        "iterations": 0,
        "should_retry": False,
        "max_iterations": 2,
    }
    
    result = agent._evaluate(state)
    assert result["iterations"] == 1
    # Low confidence and first iteration should trigger retry
    assert result["should_retry"] is True


def test_evaluate_node_max_iterations(mock_services):
    """Test evaluate node at max iterations."""
    agent = RAGAgent()
    
    state: RAGState = {
        "query": "test",
        "rephrased_query": "test",
        "file_id": None,
        "source_type": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "Short",
        "confidence": 0.0,
        "iterations": 1,
        "should_retry": False,
        "max_iterations": 2,
    }
    
    result = agent._evaluate(state)
    assert result["iterations"] == 2
    # At max iterations, should not retry
    assert result["should_retry"] is False


def test_decide_next(mock_services):
    """Test decision logic."""
    agent = RAGAgent()
    
    state_retry: RAGState = {
        "query": "test",
        "rephrased_query": None,
        "file_id": None,
        "source_type": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "confidence": 0.0,
        "iterations": 0,
        "should_retry": True,
        "max_iterations": 2,
    }
    
    assert agent._decide_next(state_retry) == "retry"
    
    state_end: RAGState = {
        "query": "test",
        "rephrased_query": None,
        "file_id": None,
        "source_type": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "confidence": 0.0,
        "iterations": 0,
        "should_retry": False,
        "max_iterations": 2,
    }
    
    assert agent._decide_next(state_end) == "end"

