"""Property layout of the default document collection."""

from weaviate.classes.config import DataType, Property


DEFAULT_COLLECTION_DESCRIPTION = (
    "Default collection for general document storage and retrieval"
)

DEFAULT_COLLECTION_PROPERTIES = [
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
        name="domain",
        data_type=DataType.TEXT,
        description="Knowledge domain the document chunk belongs to",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="date_created",
        data_type=DataType.DATE,
        description="Date and time the document chunk was created",
        index_filterable=True,
        index_searchable=False,
    ),
]
