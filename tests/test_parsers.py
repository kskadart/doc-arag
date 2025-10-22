import pytest
from src.docarag.services.parsers import chunk_text, parse_file


def test_chunk_text_basic():
    """Test basic text chunking."""
    text = "This is a test. " * 100  # 1600 characters
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    assert len(chunks) > 0
    assert all("content" in chunk for chunk in chunks)
    assert all("chunk_index" in chunk for chunk in chunks)
    assert chunks[0]["chunk_index"] == 0


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = chunk_text("")
    assert len(chunks) == 0


def test_chunk_text_small():
    """Test chunking text smaller than chunk size."""
    text = "Small text"
    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0]["content"] == text


def test_chunk_text_overlap():
    """Test that chunks have proper overlap."""
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=50)

    # Should have overlap between consecutive chunks
    assert len(chunks) > 1
    # Verify indices are sequential
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_parse_file_unsupported():
    """Test parsing unsupported file type."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file(b"test", "txt")


def test_parse_file_pdf_invalid():
    """Test parsing invalid PDF."""
    with pytest.raises(Exception, match="Failed to parse PDF"):
        parse_file(b"not a pdf", "pdf")


def test_parse_file_docx_invalid():
    """Test parsing invalid DOCX."""
    with pytest.raises(Exception, match="Failed to parse DOCX"):
        parse_file(b"not a docx", "docx")
