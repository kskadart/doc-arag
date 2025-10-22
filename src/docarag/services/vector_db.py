import logging
from typing import List, Dict, Any
from weaviate.classes.config import Configure, DataType, Property
from weaviate.collections import Collection
from weaviate.collections.classes.config import CollectionConfig
from weaviate.exceptions import WeaviateInsertManyAllFailedError

from src.docarag.clients import get_vector_db_client

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
        collection: Collection = await client.collections.use(collection_name)
        with collection.batch.fixed_size(batch_size=200) as batch:
            for obj in content_list:
                properties: Dict[str, Any] = obj["properties"]
                vector: Dict[str, List[float]] = obj["vector"]
                batch.add_object(
                    properties=properties,
                    vector=vector,
                )
                if batch.number_errors > 3:
                    logger.error("Batch import stopped due to excessive errors.")
                    break
        failed_objects = collection.batch.failed_object
        if failed_objects:
            logger.error(f"Number of failed imports: {len(failed_objects)}")
            logger.error(f"First failed object: {failed_objects[0]}")
            raise WeaviateInsertManyAllFailedError(
                f"Failed to add batch objects to collection '{collection_name}': {failed_objects}"
            )
        else:
            logger.info(f"Vectors added to collection {collection_name} successfully")
