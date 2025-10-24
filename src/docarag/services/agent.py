"""LangGraph RAG agent for multi-step document retrieval and question answering."""

import logging
from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

from src.docarag.clients.vector_db_client import get_vector_db_client
from src.docarag.clients.embedding import EmbeddingGRPCClient
from src.docarag.models.requests import QueryRequest
from src.docarag.models.responses import AgentQueryResponse
from src.docarag.settings import settings

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """State schema for the RAG agent graph."""

    query: str = Field(..., description="Original user query")
    rephrased_query: Optional[str] = Field(None, description="Optimized query for retrieval")
    query_embedding: Optional[List[float]] = Field(None, description="Vector embedding of query")
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    answer: Optional[str] = Field(None, description="Generated answer")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    iterations: int = Field(0, ge=0, description="Current iteration count")
    should_iterate: bool = Field(False, description="Whether to iterate again")
    file_id: Optional[str] = Field(None, description="Optional filter for specific document")
    max_iterations: int = Field(2, ge=1, le=5, description="Maximum retry attempts")


async def rephrase_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Rephrase the user query to optimize it for retrieval.
    
    Uses Claude to reformulate the query for better semantic search results.
    """
    query = state.query
    iterations = state.iterations
    
    logger.info(f"Rephrasing query (iteration {iterations}): {query}")
    
    llm = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0.3,
    )
    
    rephrase_prompt = f"""You are a query optimization assistant. Your task is to rephrase the user's question to make it more effective for semantic search in a document database.

User Query: {query}

Rephrase this query to be more specific, clear, and optimized for finding relevant information in technical documents. Keep it concise and focused on the key information needs.

IMPORTANT: Maintain the SAME LANGUAGE as the original query. Do not translate.

Rephrased Query:"""
    
    response = await llm.ainvoke(rephrase_prompt)
    rephrased_query = response.content.strip()
    
    logger.info(f"Rephrased query: {rephrased_query}")
    
    return {"rephrased_query": rephrased_query}


async def embed_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate embedding for the rephrased query using the embedding service.
    """
    query_text = state.rephrased_query or state.query
    
    logger.info(f"Generating embedding for query: {query_text}")
    
    embedding_client = EmbeddingGRPCClient()
    query_embedding = await embedding_client.embed_text_async(query_text)
    
    logger.info(f"Generated embedding with dimension: {len(query_embedding)}")
    
    return {"query_embedding": query_embedding}


async def retrieve_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve relevant documents from Weaviate using vector similarity search.
    """
    query_embedding = state.query_embedding
    file_id = state.file_id
    
    logger.info(f"Retrieving documents with k={settings.initial_retrieval_k}")
    
    async with get_vector_db_client() as client:
        collection = client.collections.get("DefaultDocuments")
        
        if file_id:
            logger.info(f"Filtering by file_id: {file_id}")
            response = await collection.query.near_vector(
                near_vector=query_embedding,
                limit=settings.initial_retrieval_k,
                return_metadata=["distance"],
                filters={"path": ["document_name"], "operator": "Equal", "valueText": file_id}
            )
        else:
            response = await collection.query.near_vector(
                near_vector=query_embedding,
                limit=settings.initial_retrieval_k,
                return_metadata=["distance"],
            )
        
        retrieved_docs = []
        for obj in response.objects:
            retrieved_docs.append({
                "uuid": str(obj.uuid),
                "content": obj.properties.get("content", ""),
                "document_name": obj.properties.get("document_name", ""),
                "page": obj.properties.get("page", 0),
                "date_created": obj.properties.get("date_created"),
                "distance": obj.metadata.distance if obj.metadata else None,
                "similarity_score": 1.0 - obj.metadata.distance if obj.metadata and obj.metadata.distance else 0.0,
            })
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        return {"retrieved_docs": retrieved_docs}


async def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate an answer using Claude based on the retrieved documents.
    """
    query = state.query
    retrieved_docs = state.retrieved_docs
    
    logger.info(f"Generating answer for query: {query}")
    
    if not retrieved_docs:
        logger.warning("No documents retrieved, generating fallback answer")
        return {
            "answer": "I couldn't find any relevant information in the documents to answer your question.",
            "confidence": 0.0,
        }
    
    context_parts = []
    for idx, doc in enumerate(retrieved_docs[:5], 1):
        context_parts.append(
            f"Document {idx} (from {doc['document_name']}, page {doc['page']}):\n{doc['content']}\n"
        )
    
    context = "\n".join(context_parts)
    
    llm = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=settings.anthropic_temperature,
    )
    
    generation_prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

Context from documents:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, acknowledge this and provide what information is available.

IMPORTANT: Answer in the SAME LANGUAGE as the user's question. Do not translate the question or answer to another language.

Answer:"""
    
    response = await llm.ainvoke(generation_prompt)
    answer = response.content.strip()
    
    logger.info(f"Generated answer of length: {len(answer)}")
    
    return {"answer": answer}


async def evaluate_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluate the quality of the generated answer and decide if iteration is needed.
    """
    answer = state.answer
    query = state.query
    iterations = state.iterations
    max_iterations = state.max_iterations
    retrieved_docs = state.retrieved_docs
    
    logger.info(f"Evaluating answer (iteration {iterations}/{max_iterations})")
    
    if not retrieved_docs:
        return {
            "confidence": 0.0,
            "should_iterate": False,
        }
    
    llm = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0.1,
    )
    
    evaluation_prompt = f"""You are an answer quality evaluator. Assess how well the given answer addresses the user's question.

User Question: {query}

Answer: {answer}

Evaluate the answer on a scale from 0.0 to 1.0 based on:
- Relevance to the question
- Completeness of the answer
- Use of specific information from the context

Respond with ONLY a number between 0.0 and 1.0, nothing else.

Confidence Score:"""
    
    response = await llm.ainvoke(evaluation_prompt)
    
    try:
        confidence = float(response.content.strip())
        confidence = max(0.0, min(1.0, confidence))
    except ValueError:
        logger.warning(f"Could not parse confidence score: {response.content}")
        confidence = 0.5
    
    logger.info(f"Evaluated confidence: {confidence}")
    
    should_iterate = (
        iterations < max_iterations
        and confidence < settings.agent_confidence_threshold
        and len(retrieved_docs) > 0
    )
    
    return {
        "confidence": confidence,
        "should_iterate": should_iterate,
        "iterations": iterations + 1,
    }


def should_continue(state: AgentState) -> Literal["rephrase_query", "end"]:
    """
    Determine if the agent should iterate or end.
    """
    if state.should_iterate:
        logger.info("Confidence below threshold, iterating...")
        return "rephrase_query"
    else:
        logger.info("Ending agent execution")
        return "end"


def build_agent_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent workflow.
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("rephrase_query", rephrase_query_node)
    workflow.add_node("embed_query", embed_query_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate_answer", evaluate_answer_node)
    
    workflow.set_entry_point("rephrase_query")
    
    workflow.add_edge("rephrase_query", "embed_query")
    workflow.add_edge("embed_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("generate_answer", "evaluate_answer")
    
    workflow.add_conditional_edges(
        "evaluate_answer",
        should_continue,
        {
            "rephrase_query": "rephrase_query",
            "end": END,
        },
    )
    
    return workflow.compile()


async def query_documents(request: QueryRequest) -> AgentQueryResponse:
    """
    Main entry point for querying documents using the RAG agent.
    
    Builds the agent graph, executes it with the query, and returns a structured response with generated answer.
    """
    logger.info(f"Processing query: {request.query}")
    
    initial_state = AgentState(
        query=request.query,
        file_id=None,
        max_iterations=request.max_iterations,
    )
    
    agent = build_agent_graph()
    
    final_state = await agent.ainvoke(initial_state.model_dump())
    
    return AgentQueryResponse(
        query=request.query,
        answer=final_state.get("answer", "Unable to generate an answer."),
        rephrased_query=final_state.get("rephrased_query"),
        confidence=final_state.get("confidence", 0.0),
        iterations=final_state.get("iterations", 0),
        sources_used=len(final_state.get("retrieved_docs", [])),
    )

