SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "text/markdown": "md",
    "text/plain": "md",
}

# python-magic reports plain text for markdown files, both reach the same parser
MARKDOWN_MIME_TYPES = ("text/markdown", "text/plain")

DEFAULT_COLLECTION_NAME = "DefaultDocuments"

# Knowledge domain a document belongs to; attribution and an optional query filter
DEFAULT_DOMAIN = "general"
DOMAIN_PATTERN = r"^[a-z0-9][a-z0-9-]*$"
DOMAIN_MAX_LENGTH = 64

# API embedders accept thousands of tokens; a chunk this long is a sign of a
# section that should have been split by headers, not a hard limit
MD_CHUNK_WARNING_THRESHOLD = 4000

# Conversational sessions: one Weaviate object per chat message, no vectors.
# Session ids come from the client (base36 or uuid), so they are validated as
# a single URL path segment; Latin only.
SESSION_COLLECTION_NAME = "ChatMessages"
SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
SESSION_ID_MAX_LENGTH = 64
SESSION_SUMMARY_ROLE = "summary"
