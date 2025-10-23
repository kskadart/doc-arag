from typing import List, Dict
import io
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_pdf(file_content: bytes) -> Dict[int, str]:
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

        pages_dict = {}
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                pages_dict[page_num] = text

        return pages_dict

    except Exception as e:
        raise Exception(f"Failed to parse PDF: {str(e)}")


def parse_docx(file_content: bytes) -> Dict[int, str]:
    """
    Extract text from DOCX file with section numbers.

    Args:
        file_content: DOCX file content as bytes

    Returns:
        Dictionary mapping section numbers to extracted text

    Raises:
        Exception: If DOCX parsing fails
    """
    try:
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)

        pages_dict = {}
        current_section_text = []

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                current_section_text.append(text)

        # Combine all paragraphs into a single section
        # For DOCX, we'll treat the entire document as page 1
        if current_section_text:
            pages_dict[1] = "\n\n".join(current_section_text)

        return pages_dict

    except Exception as e:
        raise Exception(f"Failed to parse DOCX: {str(e)}")


def parse_document(
    file_content: bytes, content_type: str, chunk_size: int, chunk_overlap: int
) -> List[Dict[str, str | int]]:
    """
    Parse document and split into chunks with page tracking.

    Args:
        file_content: File content as bytes
        content_type: MIME content type (e.g., "application/pdf")
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of dictionaries with chunk content and page number:
        [{"content": chunk_text, "page": page_number}, ...]

    Raises:
        ValueError: If content type is not supported
        Exception: If parsing fails
    """
    # Determine parser based on content_type
    if content_type == "application/pdf":
        pages_dict = parse_pdf(file_content)
    elif content_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        pages_dict = parse_docx(file_content)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split text and track pages
    chunks = []
    for page_num, page_text in pages_dict.items():
        page_chunks = text_splitter.split_text(page_text)
        for chunk_text in page_chunks:
            chunks.append({"content": chunk_text, "page": page_num})

    return chunks
