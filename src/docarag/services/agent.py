"""LangGraph RAG agent for multi-step document retrieval and question answering."""

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from weaviate.classes.query import Filter, MetadataQuery

from src.docarag.clients.vector_db_client import get_vector_db_client
from src.docarag.consts import DEFAULT_COLLECTION_NAME, DEFAULT_DOMAIN
from src.docarag.errors import RerankerError
from src.docarag.models.requests import QueryRequest
from src.docarag.models.responses import AgentQueryResponse, SourceChunk
from src.docarag.services.embeddings import get_embedding_service
from src.docarag.services.llm import get_chat_model
from src.docarag.services.reranker import get_reranker_service
from src.docarag.settings import settings

logger = logging.getLogger(__name__)

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")

GENERATION_SYSTEM_PROMPT = """Ты — ассистент оператора контакт-центра интернет-провайдера «Эра-Телеком». Отвечаешь на вопросы операторов по внутренней базе знаний.

Правила:
- Отвечай ТОЛЬКО на русском языке, независимо от языка контекста.
- Используй только факты из контекста. Не додумывай цены, суммы, сроки, фамилии и названия, которых там нет.
- Отвечай полно: приведи ВСЕ относящиеся к вопросу шаги, условия, сроки, названия модулей, вкладок, кнопок, подразделений и ролей, которые есть в контексте. Лучше лишняя деталь из базы знаний, чем пропущенная.
- Для процедур перечисляй шаги по порядку, нумерованным списком, ничего не пропуская.
- Если точных данных нет (например, конкретной стоимости или фамилии), прямо скажи, что в базе знаний их нет, и ОБЯЗАТЕЛЬНО добавь то, что база знаний говорит по этому вопросу: как это устроено, где посмотреть, кто отвечает, какой речевой модуль использовать. Не называй никаких сумм и фамилий.
- Сохраняй формулировки базы знаний: названия модулей, вкладок, подразделений, ролей, шагов и речевые модули приводи дословно.
- Не пиши вступлений вроде «На основании предоставленных документов» и не ссылайся на «Фрагмент N» — сразу давай ответ."""


class AgentState(BaseModel):
    """State schema for the RAG agent graph."""

    query: str = Field(..., description="Original user query")
    rephrased_query: str | None = Field(
        default=None, description="Optimized query for retrieval"
    )
    query_embedding: list[float] | None = Field(
        default=None, description="Vector embedding of query"
    )
    retrieved_docs: list[dict[str, Any]] = Field(
        default_factory=list, description="Retrieved documents"
    )
    answer: str | None = Field(default=None, description="Generated answer")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score"
    )
    iterations: int = Field(default=0, ge=0, description="Current iteration count")
    should_iterate: bool = Field(default=False, description="Whether to iterate again")
    file_id: str | None = Field(
        default=None, description="Optional filter for specific document"
    )
    domain: str | None = Field(
        default=None, description="Optional filter on the chunk knowledge domain"
    )
    max_iterations: int = Field(
        default=2, ge=1, le=5, description="Maximum retry attempts"
    )


async def rephrase_query_node(state: AgentState) -> dict[str, Any]:
    """
    Rephrase the user query to optimize it for retrieval.

    Uses the configured LLM to reformulate the query for better semantic search results.
    """
    query = state.query
    iterations = state.iterations

    logger.info(f"Rephrasing query (iteration {iterations}): {query}")

    llm = get_chat_model(0.3)

    rephrase_prompt = f"""You are a query optimization assistant. Your task is to rephrase the user's question to make it more effective for semantic search in a document database.

User Query: {query}

Rephrase this query to be more specific, clear, and optimized for finding relevant information in technical documents. Keep it concise and focused on the key information needs.

IMPORTANT: Maintain the SAME LANGUAGE as the original query. Do not translate.

Rephrased Query:"""

    response = await llm.ainvoke(rephrase_prompt)
    rephrased_query = response.text.strip()

    logger.info(f"Rephrased query: {rephrased_query}")

    return {"rephrased_query": rephrased_query}


async def embed_query_node(state: AgentState) -> dict[str, Any]:
    """
    Generate embedding for the rephrased query using the embedding service.
    """
    query_text = state.rephrased_query or state.query

    logger.info(f"Generating embedding for query: {query_text}")

    query_embedding = await get_embedding_service().embed_text_async(query_text)

    logger.info(f"Generated embedding with dimension: {len(query_embedding)}")

    return {"query_embedding": query_embedding}


def _build_retrieval_filter(file_id: str | None, domain: str | None) -> Any:
    """Combine the optional document and domain filters, `None` when both are unset."""
    filters = None
    if file_id:
        filters = Filter.by_property("document_name").equal(file_id)
    if domain:
        domain_filter = Filter.by_property("domain").equal(domain)
        filters = domain_filter if filters is None else filters & domain_filter
    return filters


async def retrieve_documents_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve relevant documents from Weaviate using vector similarity search.
    """
    query_embedding = state.query_embedding
    filters = _build_retrieval_filter(state.file_id, state.domain)

    logger.info(
        f"Retrieving documents with k={settings.initial_retrieval_k}"
        + (f", domain={state.domain}" if state.domain else "")
        + (f", file_id={state.file_id}" if state.file_id else "")
    )

    async with get_vector_db_client() as client:
        collection = client.collections.get(DEFAULT_COLLECTION_NAME)

        response = await collection.query.near_vector(
            near_vector=query_embedding,
            limit=settings.initial_retrieval_k,
            target_vector="content_vector",
            return_metadata=MetadataQuery(distance=True),
            filters=filters,
        )

        retrieved_docs = []
        for obj in response.objects:
            distance = obj.metadata.distance if obj.metadata else None
            retrieved_docs.append(
                {
                    "uuid": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "document_name": obj.properties.get("document_name", ""),
                    "page": obj.properties.get("page", 0),
                    "domain": obj.properties.get("domain", DEFAULT_DOMAIN),
                    "date_created": obj.properties.get("date_created"),
                    "distance": distance,
                    "similarity_score": 1.0 - distance if distance is not None else 0.0,
                }
            )

        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        return {"retrieved_docs": retrieved_docs}


async def rerank_documents_node(state: AgentState) -> dict[str, Any]:
    """
    Rerank retrieved documents by relevance using the configured reranker.

    Falls back to the original retrieval order, truncated to `rerank_top_k`, when
    the reranker is disabled, unavailable or answers badly, so the pipeline never
    fails on this step.
    """
    query = state.rephrased_query or state.query
    retrieved_docs = state.retrieved_docs

    if settings.reranker_provider == "none":
        logger.info("Reranker disabled, keeping retrieval order")
        return {"retrieved_docs": retrieved_docs[: settings.rerank_top_k]}

    logger.info(f"Reranking {len(retrieved_docs)} retrieved documents")

    try:
        reranked_docs = await get_reranker_service().rerank_async(
            query, retrieved_docs, top_k=settings.rerank_top_k
        )
        logger.info(f"Reranked documents, kept top {len(reranked_docs)}")
        return {"retrieved_docs": reranked_docs}
    except RerankerError as exc:
        logger.warning(
            f"Reranker service unavailable, falling back to retrieval order: {exc}"
        )
        return {"retrieved_docs": retrieved_docs[: settings.rerank_top_k]}


async def generate_answer_node(state: AgentState) -> dict[str, Any]:
    """
    Generate an answer using the configured LLM based on the retrieved documents.
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
    for idx, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"[Фрагмент {idx}, источник {doc['document_name']}, раздел {doc['page']}]\n{doc['content']}\n"
        )

    context = "\n".join(context_parts)

    llm = get_chat_model(settings.llm_temperature)

    generation_prompt = f"""Контекст из базы знаний (фрагменты, лучшие первыми):
{context}

Вопрос оператора: {query}

Ответ:"""

    response = await llm.ainvoke(
        [
            SystemMessage(content=GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=generation_prompt),
        ]
    )
    answer = response.text.strip()

    logger.info(f"Generated answer of length: {len(answer)}")

    return {"answer": answer}


def parse_confidence(text: str) -> float | None:
    """Extract the first number from an evaluator reply and clamp it to [0, 1]."""
    match = _NUMBER_PATTERN.search(text)
    if match is None:
        return None
    return max(0.0, min(1.0, float(match.group())))


async def evaluate_answer_node(state: AgentState) -> dict[str, Any]:
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

    llm = get_chat_model(0.1)

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

    confidence = parse_confidence(response.text)
    if confidence is None:
        logger.warning(f"Could not parse confidence score: {response.text}")
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


def build_agent_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """
    Build and compile the LangGraph agent workflow.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("rephrase_query", rephrase_query_node)
    workflow.add_node("embed_query", embed_query_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("rerank_documents", rerank_documents_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate_answer", evaluate_answer_node)

    workflow.set_entry_point("rephrase_query")

    workflow.add_edge("rephrase_query", "embed_query")
    workflow.add_edge("embed_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "rerank_documents")
    workflow.add_edge("rerank_documents", "generate_answer")
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
        domain=request.domain,
        max_iterations=request.max_iterations,
    )

    agent = build_agent_graph()

    final_state = await agent.ainvoke(initial_state)
    retrieved_docs = final_state.get("retrieved_docs", [])

    return AgentQueryResponse(
        query=request.query,
        answer=final_state.get("answer", "Unable to generate an answer."),
        rephrased_query=final_state.get("rephrased_query"),
        confidence=final_state.get("confidence", 0.0),
        iterations=final_state.get("iterations", 0),
        sources_used=len(retrieved_docs),
        sources=[build_source_chunk(doc) for doc in retrieved_docs],
    )


def build_source_chunk(doc: dict[str, Any], snippet_length: int = 200) -> SourceChunk:
    """Expose a retrieved chunk to API clients without the full content."""
    score = doc.get("rerank_score", doc.get("similarity_score"))
    content = str(doc.get("content", ""))
    return SourceChunk(
        document_name=str(doc.get("document_name", "")),
        domain=str(doc.get("domain", DEFAULT_DOMAIN)),
        page=int(doc.get("page") or 0),
        score=float(score) if score is not None else None,
        snippet=content[:snippet_length],
    )
