import io
import logging

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

from src.docarag.consts import MARKDOWN_MIME_TYPES, MD_CHUNK_WARNING_THRESHOLD
from src.docarag.settings import settings

logger = logging.getLogger(__name__)

MD_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
MD_BREADCRUMB_SEPARATOR = " > "
# Floor for the sub-split budget that is left after the breadcrumb prefix
MD_MIN_CHUNK_SIZE = 100


def parse_pdf(file_content: bytes) -> list[Document]:
    """
    Extract text from PDF file with page numbers.

    Args:
        file_content: PDF file content as bytes

    Returns:
        Dictionary mapping page numbers to extracted text

    Raises:
        Exception: If PDF parsing fails
    """
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)

        documents: list[Document] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text().strip()
            if len(text) > 0:
                documents.append(
                    Document(page_content=text, metadata={"page": page_num})
                )
            # if (
            #     page_num >= 7
            # ):  # TODO: Remove this once we have a better way to handle large PDFs
            #     break
        return documents

    except Exception as e:
        raise Exception(f"Failed to parse PDF: {str(e)}")


def parse_docx(file_content: bytes) -> list[Document]:
    """
    Extract text from DOCX file with section numbers.

    Args:
        file_content: DOCX file content as bytes

    Returns:
        List of documents, one per section, with the section number in "page"

    Raises:
        Exception: If DOCX parsing fails
    """
    raise NotImplementedError("DOCX parsing is not implemented yet")
    # try:
    #     docx_file = io.BytesIO(file_content)
    #     doc = Document(docx_file)

    #     pages_dict = {}
    #     current_section_text = []

    #     for paragraph in doc.paragraphs:
    #         text = paragraph.text.strip()
    #         if text:
    #             current_section_text.append(text)

    #     # Combine all paragraphs into a single section
    #     # For DOCX, we'll treat the entire document as page 1
    #     if current_section_text:
    #         combined_text = "\n\n".join(current_section_text)
    #         # cleaned_text = clean_text(combined_text)
    #         if combined_text:
    #             pages_dict[1] = combined_text

    #     return pages_dict

    # except Exception as e:
    #     raise Exception(f"Failed to parse DOCX: {str(e)}")


def _strip_frontmatter(text: str) -> str:
    """
    Drop a leading YAML frontmatter block from markdown text.

    Args:
        text: Raw markdown text

    Returns:
        Markdown body without the frontmatter block
    """
    if not text.startswith("---"):
        return text

    closing = text.find("\n---", len("---"))
    if closing == -1:
        return text

    end_of_line = text.find("\n", closing + 1)
    return text[end_of_line + 1 :] if end_of_line != -1 else ""


def _breadcrumb(metadata: dict[str, str]) -> str:
    """
    Build an "H1 > H2" trail from the header metadata of a markdown section.

    Args:
        metadata: Header metadata produced by MarkdownHeaderTextSplitter

    Returns:
        Breadcrumb string, empty when the section carries no headers
    """
    titles = [metadata[key] for _, key in MD_HEADERS_TO_SPLIT_ON if metadata.get(key)]
    return MD_BREADCRUMB_SEPARATOR.join(titles)


def parse_markdown(
    file_content: bytes, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """
    Extract text from markdown file split by headers into numbered sections.

    Frontmatter is dropped, sections are split on h1-h3 and oversized ones are
    split further; every chunk keeps the breadcrumb of its section so that a
    continuation chunk still carries the context of its headers.

    Args:
        file_content: Markdown file content as bytes
        chunk_size: Maximum size of a chunk in characters, breadcrumb included
        chunk_overlap: Number of characters to overlap between sub-chunks

    Returns:
        List of documents, one per chunk, with the section number in "page"

    Raises:
        Exception: If markdown parsing fails
    """
    try:
        text = file_content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise Exception(f"Failed to parse markdown: {str(e)}")

    body = _strip_frontmatter(text)
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=MD_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    ).split_text(body)
    if not sections:
        sections = [Document(page_content=body)]

    documents: list[Document] = []
    for section_number, section in enumerate(sections, start=1):
        breadcrumb = _breadcrumb(section.metadata)
        prefix = f"{breadcrumb}\n\n" if breadcrumb else ""
        budget = max(chunk_size - len(prefix), MD_MIN_CHUNK_SIZE)

        if len(section.page_content) <= budget:
            pieces = [section.page_content]
        else:
            pieces = RecursiveCharacterTextSplitter(
                chunk_size=budget,
                chunk_overlap=chunk_overlap,
                length_function=len,
            ).split_text(section.page_content)

        for piece in pieces:
            if not piece.strip():
                continue
            content = f"{prefix}{piece}"
            if len(content) > MD_CHUNK_WARNING_THRESHOLD:
                logger.warning(
                    f"Markdown chunk of {len(content)} characters exceeds "
                    f"{MD_CHUNK_WARNING_THRESHOLD}, embedding may be truncated"
                )
            documents.append(
                Document(page_content=content, metadata={"page": section_number})
            )

    return documents


def parse_document(
    file_content: bytes, content_type: str, chunk_size: int, chunk_overlap: int
) -> list[dict[str, str | int]]:
    """
    Parse document and split into chunks with page tracking.

    Args:
        file_content: File content as bytes
        content_type: MIME content type (e.g., "application/pdf")
        chunk_size: Maximum size of each chunk in characters, markdown excluded
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of dictionaries with chunk content and page number:
        [{"content": chunk_text, "page": page_number}, ...]

    Raises:
        ValueError: If content type is not supported
        Exception: If parsing fails

    Note:
        Markdown ignores the caller-supplied sizes and uses settings.md_chunk_size
        and settings.md_chunk_overlap: header-scoped sections need a larger budget
        than the page-based split the other formats go through.
    """

    if content_type in MARKDOWN_MIME_TYPES:
        markdown_documents = parse_markdown(
            file_content,
            chunk_size=settings.md_chunk_size,
            chunk_overlap=settings.md_chunk_overlap,
        )
        return [
            {
                "content": document.page_content,
                "page": document.metadata["page"],
            }
            for document in markdown_documents
        ]

    if content_type == "application/pdf":
        documents = parse_pdf(file_content)
    elif content_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        # TODO: NOT IMPLEMENTED YET
        documents = parse_docx(file_content)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=True,
    )
    documents = text_splitter.split_documents(documents)
    chunks = [
        {
            "content": document.page_content,
            "page": document.metadata["page"],
        }
        for document in documents
    ]

    return chunks
