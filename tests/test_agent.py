"""Tests for the LangGraph RAG agent."""

from src.docarag.services.agent import (
    AgentState,
    should_continue,
)


def test_should_continue():
    """Test conditional routing logic based on agent state."""
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

