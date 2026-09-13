import logging
from typing import Any

from weaviate.classes.config import Configure
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.collections.classes.config import CollectionConfig
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.exceptions import WeaviateInsertManyAllFailedError

from src.docarag.clients import get_vector_db_client
from src.docarag.consts import DEFAULT_COLLECTION_NAME, DEFAULT_DOMAIN
from src.docarag.errors import EmbeddingError
from src.docarag.models.responses import VectorSearchResponse, VectorSearchResult
from src.docarag.services.embeddings import get_embedding_service
from src.docarag.settings import settings
from src.docarag.utils.default_collection_conf import (
    DEFAULT_COLLECTION_DESCRIPTION,
    DEFAULT_COLLECTION_PROPERTIES,
)


logger = logging.getLogger(__name__)


async def is_collection_exists(collection_name: str) -> bool:
    async with get_vector_db_client() as client:
        if await client.collections.exists(collection_name):
            return True
        return False


async def create_default_collection() -> None:
    """Create the default document collection unless it already exists."""
    collection_name = DEFAULT_COLLECTION_NAME
    if await is_collection_exists(collection_name):
        logger.info(f"Collection {collection_name} already exists")
        return
    async with get_vector_db_client() as client:
        await client.collections.create(
            name=collection_name,
            description=DEFAULT_COLLECTION_DESCRIPTION,
            properties=DEFAULT_COLLECTION_PROPERTIES,
            vector_config=Configure.Vectors.self_provided(
                name="content_vector",
            ),
        )
        logger.info(f"Collection {collection_name} created successfully")


async def recreate_default_collection() -> None:
    """Drop the default collection (if any) and create it again with the current schema."""
    await delete_collection(DEFAULT_COLLECTION_NAME)
    await create_default_collection()


async def verify_embedding_dimension() -> int | None:
    """
    Compare the live embedding dimension with the vectors already stored.

    Returns:
        The probed dimension, `None` when the embedding endpoint is unreachable
        (the API must still boot for uploads and listings to work)

    Raises:
        RuntimeError: If stored vectors have a different dimension than the
            configured embedding model produces, i.e. the collection must be
            recreated before any new insert or query can succeed
    """
    try:
        probed = await get_embedding_service().get_embedding_dimension_async()
    except (EmbeddingError, ValueError) as exc:
        logger.warning(
            f"Embedding endpoint unavailable, skipping dimension check: {exc}"
        )
        return None

    if not await is_collection_exists(DEFAULT_COLLECTION_NAME):
        logger.info(f"Embedding dimension {probed}, collection not created yet")
        return probed

    async with get_vector_db_client() as client:
        collection = client.collections.get(DEFAULT_COLLECTION_NAME)
        response = await collection.query.fetch_objects(
            limit=1, include_vector=["content_vector"]
        )

    if not response.objects:
        logger.info(f"Embedding dimension {probed}, collection is empty")
        return probed

    stored = len(response.objects[0].vector["content_vector"])
    if stored != probed:
        raise RuntimeError(
            f"Stored vectors have dimension {stored} but embedding model "
            f"'{settings.embedding_model}' produces {probed}; recreate the "
            f"collection (scripts.load_corpus --recreate) before continuing"
        )
    logger.info(f"Embedding dimension {probed} matches stored vectors")
    return probed


async def create_collection_from_config(collection_config: CollectionConfig) -> None:
    collection_name = collection_config.name
    if await is_collection_exists(collection_name):
        logger.info(f"Collection {collection_name} already exists")
        return
    async with get_vector_db_client() as client:
        await client.collections.create_from_config(collection_config)
        logger.info(f"Collection {collection_name} created successfully")


async def delete_collection(collection_name: str) -> None:
    if not await is_collection_exists(collection_name):
        logger.info(f"Collection {collection_name} does not exist")
        return
    async with get_vector_db_client() as client:
        await client.collections.delete(collection_name)
        logger.info(f"Collection {collection_name} deleted successfully")


async def delete_objects_by_document_name(
    collection_name: str, document_name: str
) -> int:
    """
    Delete every chunk that belongs to a document.

    Keeps re-embedding idempotent and clears the vector database when a
    document is removed from storage.

    Args:
        collection_name: Name of the collection to purge
        document_name: Value of the document_name property to match

    Returns:
        Number of deleted objects, zero when the collection does not exist
    """
    if not await is_collection_exists(collection_name):
        logger.info(f"Collection {collection_name} does not exist")
        return 0

    async with get_vector_db_client() as client:
        collection = client.collections.use(collection_name)
        result = await collection.data.delete_many(
            where=Filter.by_property("document_name").equal(document_name)
        )
        deleted_count: int = result.successful
        logger.info(
            f"Deleted {deleted_count} objects of '{document_name}' "
            f"from collection {collection_name}"
        )
        return deleted_count


async def add_batch_objects(
    collection_name: str, content_list: list[dict[str, Any]]
) -> None:
    """
    Insert chunk objects with their named vector using Weaviate batch inserts.

    Raises:
        ValueError: If the collection does not exist (a silent no-op here would
            let the embedding task report success with nothing stored)
        WeaviateInsertManyAllFailedError: If any object in a batch is rejected
    """
    if not await is_collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist")

    if not content_list:
        logger.info(f"Nothing to insert into collection {collection_name}")
        return

    async with get_vector_db_client() as client:
        collection = client.collections.get(collection_name)

        batch_size = settings.weaviate_insert_batch_size
        inserted = 0
        for start in range(0, len(content_list), batch_size):
            batch = content_list[start : start + batch_size]
            objects = [
                DataObject(properties=obj["properties"], vector=obj["vector"])
                for obj in batch
            ]
            result = await collection.data.insert_many(objects)
            if result.has_errors:
                first_error = next(iter(result.errors.values()))
                raise WeaviateInsertManyAllFailedError(
                    f"Failed to add {len(result.errors)} of {len(batch)} objects to "
                    f"collection '{collection_name}': {first_error.message}"
                )
            inserted += len(batch)

        logger.info(
            f"Successfully added {inserted} vectors to collection {collection_name}"
        )


async def find_nearest_vectors(
    query: str,
    collection_name: str,
    limit: int,
) -> VectorSearchResponse:
    """
    Find nearest vectors in a collection based on text query.

    Args:
        query: Text query to search for
        collection_name: Name of the collection to search
        limit: Maximum number of results to return (defaults to settings.initial_retrieval_k)

    Returns:
        VectorSearchResponse with search results

    Raises:
        ValueError: If collection does not exist or query is empty
        Exception: If embedding or search fails
    """
    if not await is_collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist")

    logger.info(
        f"Searching for nearest vectors in collection '{collection_name}' with query: '{query[:100]}...'"
    )

    query_vector = await get_embedding_service().embed_text_async(query)
    logger.debug(f"Generated query embedding with dimension: {len(query_vector)}")

    async with get_vector_db_client() as client:
        collection = client.collections.use(collection_name)

        response = await collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            target_vector="content_vector",
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            result = VectorSearchResult(
                uuid=str(obj.uuid),
                document_name=obj.properties.get("document_name", ""),
                page=obj.properties.get("page", 0),
                content=obj.properties.get("content", ""),
                domain=obj.properties.get("domain", DEFAULT_DOMAIN),
                date_created=obj.properties.get("date_created"),
                similarity_score=(
                    1.0 - obj.metadata.distance
                    if obj.metadata.distance is not None
                    else 0.0
                ),
            )
            results.append(result)

        logger.info(
            f"Found {len(results)} nearest vectors in collection '{collection_name}'"
        )

        return VectorSearchResponse(
            query=query,
            collection_name=collection_name,
            results=results,
            total_results=len(results),
        )
