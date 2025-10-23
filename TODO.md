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
