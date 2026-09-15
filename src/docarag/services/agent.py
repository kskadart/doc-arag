"""LangGraph RAG agent for multi-step document retrieval and question answering."""

import logging
import re
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from weaviate.classes.query import Filter, MetadataQuery

from src.docarag.clients.vector_db_client import get_vector_db_client
from src.docarag.consts import DEFAULT_COLLECTION_NAME, DEFAULT_DOMAIN
from src.docarag.errors import RerankerError
from src.docarag.models.requests import QueryRequest
from src.docarag.models.responses import AgentQueryResponse, SourceChunk
from src.docarag.models.sessions import ChatTurn
from src.docarag.services.embeddings import get_embedding_service
from src.docarag.services.llm import get_chat_model
from src.docarag.services.reranker import get_reranker_service
from src.docarag.services.sessions import (
    format_history_transcript,
    history_to_messages,
    load_memory,
    record_turn,
    schedule_summary_refresh,
)
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
- Разговор может продолжать прежнюю тему: учитывай предыдущие реплики и сводку, оставайся последовательным с уже сказанным, но никогда не противоречь контексту.
- Не пиши вступлений вроде «На основании предоставленных документов» и не ссылайся на «Фрагмент N» — сразу давай ответ."""


class AgentState(BaseModel):
    """State schema for the RAG agent graph."""

    query: str = Field(..., description="Original user query")
    rephrased_query: str | None = Field(
        default=None,
        description=(
            "Standalone query resolved against the conversation; this is what "
            "gets embedded, reranked and evaluated"
        ),
    )
    session_id: str | None = Field(
        default=None, description="Chat the query belongs to"
    )
    history: list[ChatTurn] = Field(
        default_factory=list, description="Verbatim tail of the conversation"
    )
    history_summary: str | None = Field(
        default=None, description="Rolling summary of turns older than the tail"
    )
    previous_queries: list[str] = Field(
        default_factory=list,
        description="Standalone queries already tried in this run",
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


CONDENSE_SYSTEM_PROMPT = """You rewrite an operator's latest message into ONE standalone search query for a document retrieval system used by customer-support operators.

Rules:
- Resolve pronouns, ellipsis and references using the conversation below ("how much does it cost?" -> "cost of <the service discussed>").
- If the latest message asks about a previous answer ("repeat the second point", "explain that in more detail"), build the query from the TOPIC of that answer.
- If the message is already self-contained, only optimise it for semantic search: specific, key nouns, service names, document terms.
- Keep it short and focused on the key information need.
- Do NOT answer the question, do NOT explain, do NOT add quotes or prefixes.
- Write the query in the SAME LANGUAGE as the latest message. Never translate.
Output the query and nothing else."""


def build_condense_prompt(state: AgentState) -> str:
    """Human part of the condense prompt: summary, transcript, latest message."""
    transcript = format_history_transcript(
        state.history, settings.session_message_max_chars
    )
    parts = [
        f"Conversation summary so far:\n{state.history_summary or '(none)'}",
        f"Recent conversation:\n{transcript or '(none)'}",
        f"Latest operator message:\n{state.query}",
    ]
    if state.previous_queries:
        tried = "\n".join(
            f"{idx}. {query}" for idx, query in enumerate(state.previous_queries, 1)
        )
        parts.append(
            "These search queries were already tried and the answer was not good "
            f"enough:\n{tried}\n\n"
            "Produce a DIFFERENT formulation of the same information need: use "
            "synonyms, spell out abbreviations, add the domain term an internal "
            "document would use, or narrow to the single most important part of "
            "the question. Do not repeat any query listed above.\n\n"
            "Alternative standalone search query:"
        )
    else:
        parts.append("Standalone search query:")
    return "\n\n".join(parts)


async def rephrase_query_node(state: AgentState) -> dict[str, Any]:
    """
    Turn the latest message into a standalone query for retrieval.

    Follow-ups are resolved against the conversation; on a retry the LLM is
    asked for a formulation different from the ones already tried.
    """
    logger.info(f"Condensing query (iteration {state.iterations}): {state.query}")

    llm = get_chat_model(0.3)
    response = await llm.ainvoke(
        [
            SystemMessage(content=CONDENSE_SYSTEM_PROMPT),
            HumanMessage(content=build_condense_prompt(state)),
        ]
    )
    rephrased_query = response.text.strip() or state.query

    logger.info(f"Standalone query: {rephrased_query}")

    return {
        "rephrased_query": rephrased_query,
        "previous_queries": [*state.previous_queries, rephrased_query],
    }


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

    system_prompt = GENERATION_SYSTEM_PROMPT
    if state.history_summary:
        system_prompt = (
            f"{GENERATION_SYSTEM_PROMPT}\n\n"
            f"Сводка предыдущего разговора с оператором:\n{state.history_summary}"
        )

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *history_to_messages(state.history, settings.session_message_max_chars),
        HumanMessage(content=generation_prompt),
    ]

    response = await llm.ainvoke(messages)
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
    # A follow-up like "and for companies?" only makes sense once resolved
    query = state.rephrased_query or state.query
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

    evaluation_prompt = f"""You are an answer quality evaluator. Assess how well the given answer addresses the operator's question.

Question (resolved against the conversation): {query}

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

    With a `session_id` the conversation memory is loaded first and both turns
    are persisted afterwards; without one the run is fully stateless.
    """
    logger.info(f"Processing query: {request.query}")

    memory = await load_memory(request.session_id)

    initial_state = AgentState(
        query=request.query,
        file_id=None,
        domain=request.domain,
        max_iterations=request.max_iterations,
        session_id=request.session_id,
        history=memory.recent,
        history_summary=memory.summary,
    )

    agent = build_agent_graph()

    final_state = await agent.ainvoke(initial_state)
    retrieved_docs = final_state.get("retrieved_docs", [])

    response = AgentQueryResponse(
        query=request.query,
        answer=final_state.get("answer", "Unable to generate an answer."),
        rephrased_query=final_state.get("rephrased_query"),
        confidence=final_state.get("confidence", 0.0),
        iterations=final_state.get("iterations", 0),
        sources_used=len(retrieved_docs),
        sources=[build_source_chunk(doc) for doc in retrieved_docs],
        session_id=request.session_id,
    )

    if await record_turn(request.session_id, memory, request.query, response):
        schedule_summary_refresh(request.session_id, memory)

    return response


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
