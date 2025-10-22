from contextlib import asynccontextmanager
import logging
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import weaviate
from weaviate.exceptions import WeaviateConnectionError
from src.docarag.settings import settings


logger = logging.getLogger(__name__)


async def check_vector_db_connection() -> None:
    """Check if the vector database is connected with retries."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(WeaviateConnectionError),
    ):
        with attempt:
            try:
                async with get_vector_db_client() as client:
                    if not await client.is_ready():
                        raise WeaviateConnectionError("Weaviate client is not ready")
                    logger.info("Vector database connection successful")
            except WeaviateConnectionError as exc:
                raise WeaviateConnectionError(f"Failed to connect to Weaviate: {exc}")


@asynccontextmanager
async def get_vector_db_client() -> weaviate.WeaviateAsyncClient:
    async with weaviate.use_async_with_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
    ) as client:
        yield client
