from weaviate.collections.classes.config import (
    BM25Config,
    CollectionConfig,
    InvertedIndexConfig,
    ReferencePropertyConfig,
    StopwordsConfig,
    StopwordsPreset,
    Configure,
    Property,
    DataType,
    ReplicationDeletionStrategy,
)


DEFAULT_COLLECTION_CONFIG = CollectionConfig(
    name="DefaultDocuments",
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
    ],
    inverted_index_config=InvertedIndexConfig(
        bm25=BM25Config(
            b=0.75,
            k1=1.2,
        ),
        cleanup_interval_seconds=60,
        index_property_length=True,
        index_null_state=True,
        index_timestamps=True,
        stopwords=StopwordsConfig(
            preset=StopwordsPreset.NONE,
            additions=[],
            removals=[],
        ),
    ),
    vector_config=Configure.NamedVectors.none(
        name="content_vector",
        source_properties=["content"],
    ),
    multi_tenancy_config=Configure.multi_tenancy(
        enabled=False,
    ),
    sharding_config=Configure.sharding(
        virtual_per_physical=128,
        desired_count=1,
        desired_virtual_count=128,
    ),
    replication_config=Configure.replication(
        factor=1,
        async_enabled=True,
        deletion_strategy=ReplicationDeletionStrategy.TIME_BASED_RESOLUTION,
    ),
    generative_config=None,
    references=[
        ReferencePropertyConfig(
            name="document_name", description="Document name", target_collections=[]
        )
    ],
)
