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

# Knowledge domain a document belongs to; attribution only, not a search filter
DEFAULT_DOMAIN = "general"
DOMAIN_PATTERN = r"^[a-z0-9][a-z0-9-]*$"
DOMAIN_MAX_LENGTH = 64

# The embedding service silently truncates above 512 tokens; warn well before that
MD_CHUNK_WARNING_THRESHOLD = 1500
