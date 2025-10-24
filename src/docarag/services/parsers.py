from typing import List, Dict
import io
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_pdf(file_content: bytes) -> List[Document]:
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

        documents: List[Document] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text().strip()
            if len(text) > 0:
                documents.append(
                    Document(page_content=text, metadata={"page": page_num})
                )
            if (
                page_num >= 7
            ):  # TODO: Remove this once we have a better way to handle large PDFs
                break
        return documents

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
