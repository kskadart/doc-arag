from typing import List, Dict, Any
import io
from PyPDF2 import PdfReader
from docx import Document
from src.docarag.config import settings


def parse_pdf(file_content: bytes) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file_content: PDF file content as bytes
        
    Returns:
        Extracted text from all pages
        
    Raises:
        Exception: If PDF parsing fails
    """
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    except Exception as e:
        raise Exception(f"Failed to parse PDF: {str(e)}")


def parse_docx(file_content: bytes) -> str:
    """
    Extract text from DOCX file.
    
    Args:
        file_content: DOCX file content as bytes
        
    Returns:
        Extracted text from all paragraphs
        
    Raises:
        Exception: If DOCX parsing fails
    """
    try:
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)
        
        text_parts = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    except Exception as e:
        raise Exception(f"Failed to parse DOCX: {str(e)}")


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of dictionaries with chunk content and metadata
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap
    
    if not text or not text.strip():
        return []
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end].strip()
        
        if chunk_content:
            chunks.append({
                "content": chunk_content,
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": min(end, len(text)),
            })
            chunk_index += 1
        
        start = end - overlap
        
        # Avoid infinite loop if overlap >= chunk_size
        if overlap >= chunk_size:
            start = end
    
    return chunks


def parse_file(file_content: bytes, file_type: str) -> str:
    """
    Parse file based on its type.
    
    Args:
        file_content: File content as bytes
        file_type: File type (pdf, docx)
        
    Returns:
        Extracted text
        
    Raises:
        ValueError: If file type is not supported
    """
    file_type = file_type.lower()
    
    if file_type == "pdf":
        return parse_pdf(file_content)
    elif file_type in ("docx", "doc"):
        return parse_docx(file_content)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

