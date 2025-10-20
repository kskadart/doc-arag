from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from src.docarag.config import settings
from src.docarag.services.embeddings import get_embedding_service
from src.docarag.services.vectorstore import get_vectorstore_service
# from src.docarag.services.reranker import get_reranker_service


class RAGState(TypedDict):
    """State for RAG agent workflow."""
    
    query: str
    rephrased_query: Optional[str]
    file_id: Optional[str]
    source_type: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    answer: str
    confidence: float
    iterations: int
    should_retry: bool
    max_iterations: int


class RAGAgent:
    """LangGraph agent for intelligent RAG queries."""
    
    def __init__(self):
        """Initialize RAG agent with LangGraph workflow."""
        self.llm = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )
        self.embedding_service = get_embedding_service()
        self.vectorstore_service = get_vectorstore_service()
        # self.reranker_service = get_reranker_service()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("understand_query", self._understand_query)
        workflow.add_node("retrieve", self._retrieve)
        # workflow.add_node("rerank", self._rerank)
        workflow.add_node("generate", self._generate)
        workflow.add_node("evaluate", self._evaluate)
        
        # Define edges
        workflow.set_entry_point("understand_query")
        workflow.add_edge("understand_query", "retrieve")
        # workflow.add_edge("retrieve", "rerank")
        # workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "evaluate")
        
        # Conditional edge: retry or end
        workflow.add_conditional_edges(
            "evaluate",
            self._decide_next,
            {
                "retry": "understand_query",
                "end": END,
            }
        )
        
        return workflow.compile()
    
    def _understand_query(self, state: RAGState) -> RAGState:
        """
        Analyze and potentially rephrase the query for better retrieval.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with rephrased query
        """
        query = state["query"]
        
        # Use LLM to understand and rephrase query if needed
        prompt = f"""Analyze this user query and rephrase it to be more effective for document retrieval.
Focus on key concepts and make it more specific if needed.

Original Query: {query}

Provide only the rephrased query, nothing else."""
        
        try:
            response = self.llm.invoke(prompt)
            rephrased = response.content.strip()
            
            # Use rephrased query if it's different and not too long
            if rephrased and rephrased != query and len(rephrased) < 500:
                state["rephrased_query"] = rephrased
            else:
                state["rephrased_query"] = query
        
        except Exception:
            # Fallback to original query if rephrasing fails
            state["rephrased_query"] = query
        
        return state
    
    def _retrieve(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents from vector store.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with retrieved documents
        """
        query_text = state["rephrased_query"] or state["query"]
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query_text)
        
        # Search vector store
        results = self.vectorstore_service.search(
            query_vector=query_embedding,
            limit=settings.initial_retrieval_k,
            file_id=state.get("file_id"),
            source_type=state.get("source_type"),
        )
        
        state["retrieved_docs"] = results
        return state
    
    def _rerank(self, state: RAGState) -> RAGState:
        """
        Rerank retrieved documents using cross-encoder.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with reranked documents
        """
        query_text = state["rephrased_query"] or state["query"]
        retrieved_docs = state["retrieved_docs"]
        
        if not retrieved_docs:
            state["reranked_docs"] = []
            return state
        
        # Rerank documents
        reranked = self.reranker_service.rerank(
            query=query_text,
            documents=retrieved_docs,
            top_k=settings.rerank_top_k,
        )
        
        state["reranked_docs"] = reranked
        return state
    
    def _generate(self, state: RAGState) -> RAGState:
        """
        Generate answer using retrieved and reranked documents.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with generated answer
        """
        query = state["query"]
        docs = state["reranked_docs"]
        
        if not docs:
            state["answer"] = "I couldn't find any relevant information to answer your question."
            state["confidence"] = 0.0
            return state
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[Source {i}] (from {doc['filename']}, chunk {doc['chunk_index']})\n{doc['content']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.
Be accurate and cite the sources when appropriate. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Provide a clear and concise answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            state["answer"] = response.content.strip()
        except Exception as e:
            state["answer"] = f"Error generating answer: {str(e)}"
        
        return state
    
    def _evaluate(self, state: RAGState) -> RAGState:
        """
        Evaluate answer quality and decide if retry is needed.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with confidence score and retry decision
        """
        query = state["query"]
        answer = state["answer"]
        docs = state["reranked_docs"]
        
        # Calculate confidence based on multiple factors
        confidence = 0.0
        
        # Factor 1: Number of relevant documents
        if docs:
            doc_factor = min(len(docs) / settings.rerank_top_k, 1.0)
            confidence += doc_factor * 0.3
        
        # Factor 2: Rerank scores
        if docs:
            avg_rerank_score = sum(d.get("rerank_score", 0) for d in docs) / len(docs)
            # Normalize rerank score (typically between -10 and 10)
            normalized_rerank = max(0, min((avg_rerank_score + 5) / 15, 1.0))
            confidence += normalized_rerank * 0.4
        
        # Factor 3: Answer length (very short answers might indicate failure)
        if len(answer) > 50:
            confidence += 0.3
        elif len(answer) > 20:
            confidence += 0.15
        
        state["confidence"] = confidence
        
        # Decide if retry is needed
        current_iteration = state.get("iterations", 0) + 1
        max_iterations = state.get("max_iterations", 2)
        
        state["iterations"] = current_iteration
        
        # Retry if confidence is low and haven't reached max iterations
        if confidence < 0.5 and current_iteration < max_iterations:
            state["should_retry"] = True
        else:
            state["should_retry"] = False
        
        return state
    
    def _decide_next(self, state: RAGState) -> str:
        """
        Decide whether to retry or end.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name ("retry" or "end")
        """
        if state.get("should_retry", False):
            return "retry"
        return "end"
    
    async def ainvoke(
        self,
        query: str,
        file_id: Optional[str] = None,
        source_type: Optional[str] = None,
        max_iterations: int = 2
    ) -> Dict[str, Any]:
        """
        Asynchronously invoke the RAG agent.
        
        Args:
            query: User query
            file_id: Optional file filter
            source_type: Optional source type filter
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary with answer, sources, confidence, and metadata
        """
        initial_state: RAGState = {
            "query": query,
            "rephrased_query": None,
            "file_id": file_id,
            "source_type": source_type,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "should_retry": False,
            "max_iterations": max_iterations,
        }
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Format response
        return {
            "answer": final_state["answer"],
            "sources": [
                {
                    "file_id": doc["file_id"],
                    "content": doc["content"],
                    "score": doc.get("rerank_score", doc.get("score", 0)),
                    "source_type": doc["source_type"],
                    "chunk_index": doc["chunk_index"],
                }
                for doc in final_state["reranked_docs"]
            ],
            "confidence": final_state["confidence"],
            "iterations": final_state["iterations"],
            "rephrased_query": final_state["rephrased_query"],
        }
    
    def invoke(
        self,
        query: str,
        file_id: Optional[str] = None,
        source_type: Optional[str] = None,
        max_iterations: int = 2
    ) -> Dict[str, Any]:
        """
        Synchronously invoke the RAG agent.
        
        Args:
            query: User query
            file_id: Optional file filter
            source_type: Optional source type filter
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary with answer, sources, confidence, and metadata
        """
        initial_state: RAGState = {
            "query": query,
            "rephrased_query": None,
            "file_id": file_id,
            "source_type": source_type,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "should_retry": False,
            "max_iterations": max_iterations,
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Format response
        return {
            "answer": final_state["answer"],
            "sources": [
                {
                    "file_id": doc["file_id"],
                    "content": doc["content"],
                    "score": doc.get("rerank_score", doc.get("score", 0)),
                    "source_type": doc["source_type"],
                    "chunk_index": doc["chunk_index"],
                }
                for doc in final_state["reranked_docs"]
            ],
            "confidence": final_state["confidence"],
            "iterations": final_state["iterations"],
            "rephrased_query": final_state["rephrased_query"],
        }


# Global RAG agent instance
rag_agent: Optional[RAGAgent] = None


def get_rag_agent() -> RAGAgent:
    """Get or create RAG agent instance."""
    global rag_agent
    if rag_agent is None:
        rag_agent = RAGAgent()
    return rag_agent

