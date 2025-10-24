import logging
from typing import List, Dict, Any
from weaviate.classes.config import Configure, DataType, Property
from weaviate.collections.classes.config import CollectionConfig
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.exceptions import WeaviateInsertManyAllFailedError

from src.docarag.clients import get_vector_db_client
from src.docarag.clients.embedding import EmbeddingGRPCClient
from src.docarag.models.responses import VectorSearchResponse, VectorSearchResult


logger = logging.getLogger(__name__)


async def is_collection_exists(collection_name: str) -> bool:
    async with get_vector_db_client() as client:
        if await client.collections.exists(collection_name):
            return True
        return False


async def create_default_collection() -> None:
    collection_name: str = "DefaultDocuments"
    if await is_collection_exists(collection_name):
        logger.info(f"Collection {collection_name} already exists")
        return
    async with get_vector_db_client() as client:
        await client.collections.create(
            name=collection_name,
            description="Default collection for general document storage and retrieval",
            properties=[
                Property(
                    name="document_name",
                    data_type=DataType.TEXT,
                    description="Name of the document",
                    index_filterable=True,
                    index_searchable=True,
                ),
                Property(
                    name="page",
                    data_type=DataType.INT,
                    description="Page number within the document",
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="Text content of the document chunk",
                    index_searchable=True,
                ),
                Property(
                    name="date_created",
                    data_type=DataType.DATE,
                    description="Date and time the document chunk was created",
                    index_filterable=True,
                    index_searchable=False,
                ),
            ],
            vector_config=Configure.Vectors.self_provided(
                name="content_vector",
            ),
        )
        logger.info(f"Collection {collection_name} created successfully")


async def create_collection_from_config(collection_config: CollectionConfig) -> None:
    collection_name = collection_config["name"]
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


async def add_batch_objects(
    collection_name: str, content_list: List[Dict[str, Any]]
) -> None:
    if not await is_collection_exists(collection_name):
        logger.info(f"Collection {collection_name} does not exist")
        return

    async with get_vector_db_client() as client:
        collection = client.collections.get(collection_name)

        # Insert objects one by one using the async API
        failed_count = 0
        for obj in content_list:
            try:
                properties: Dict[str, Any] = obj["properties"]
                vector: Dict[str, List[float]] = obj["vector"]

                await collection.data.insert(
                    properties=properties,
                    vector=vector,
                )
            except Exception as e:
                logger.error(f"Failed to insert object: {str(e)}")
                failed_count += 1
                if failed_count > 3:
                    logger.error("Too many errors, stopping batch insert.")
                    raise WeaviateInsertManyAllFailedError(
                        f"Failed to add batch objects to collection '{collection_name}': {failed_count} failures"
                    )

        if failed_count > 0:
            logger.warning(
                f"Completed with {failed_count} failures out of {len(content_list)} objects"
            )
        else:
            logger.info(
                f"Successfully added {len(content_list)} vectors to collection {collection_name}"
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

    async with EmbeddingGRPCClient() as embedding_client:
        query_vector = await embedding_client.embed_text_async(query)
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
