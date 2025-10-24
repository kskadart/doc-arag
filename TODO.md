# TODO List

## Weaviate Connection Timeout Fix

### [x] Fixed - Disable PyPI version check in Weaviate client
- **Issue**: Weaviate client was timing out during startup trying to verify package version on PyPI
- **Solution**: Added `skip_init_checks=True` to `weaviate.use_async_with_local()` in `src/docarag/clients/vector_db_client.py`
- **Status**: ✅ Completed
- **File**: `src/docarag/clients/vector_db_client.py`

### [ ] Test application startup with Docker container
- Verify the application starts without `httpx.ConnectTimeout` errors
- Ensure Weaviate container is properly initialized
- **Status**: Pending

### [ ] Verify Weaviate is ready and accessible after startup
- Check that `client.is_ready()` returns True
- Verify connection pooling works correctly
- **Status**: Pending

### [ ] Monitor logs for any connection warnings or resource leaks
- Watch for ResourceWarning messages related to TCP connections
- Verify proper cleanup of Weaviate connections
- **Status**: Pending

## LangGraph RAG Agent

### [x] Implement multi-step RAG agent with LangGraph
- **Solution**: Created `src/docarag/services/agent.py` with stateless LangGraph workflow
- **Features**:
  - Query rephrasing using Claude for optimization
  - Embedding generation via gRPC service
  - Vector similarity search in Weaviate
  - Answer generation with context
  - Quality evaluation and iterative improvement
- **Status**: ✅ Completed
- **File**: `src/docarag/services/agent.py`

### [x] Update /query API endpoint
- **Solution**: Implemented endpoint to use the LangGraph agent
- **Status**: ✅ Completed
- **File**: `src/docarag/api.py`

### [x] Add agent configuration settings
- **Solution**: Added agent-specific settings (confidence threshold, temperature, reranker URL)
- **Status**: ✅ Completed
- **File**: `src/docarag/settings.py`

### [ ] Implement reranker client for external reranking service
- **Goal**: Create `src/docarag/clients/reranker_client.py` similar to EmbeddingGRPCClient
- **Requirements**:
  - Accept query text and list of candidate documents
  - Return top_k reranked results with scores
  - Use `rerank_top_k` from settings
  - Support both sync and async operations
- **Integration**: Add rerank node between retrieve and generate in agent graph
- **Status**: Pending (to be implemented later)
