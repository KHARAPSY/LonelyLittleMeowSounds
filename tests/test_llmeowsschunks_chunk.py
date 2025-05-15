import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from lolimeowss.main import LLMeowsSChunks

@pytest.fixture
def llmeowsschunks(tmp_path):
    """Fixture to initialize LLMeowsSChunks with a temporary directory and mocked logger."""
    llms_chunks = LLMeowsSChunks()
    llms_chunks.TMP_DIR = str(tmp_path) + "/"
    llms_chunks.logger = MagicMock()
    return llms_chunks

@pytest.fixture
def binary_file():
    """Fixture to provide a mock BinaryIO object."""
    return BytesIO(b"llmeowss content")

def test_get_chunking_content_extraction_failure(llmeowsschunks, binary_file):
    """Test handling of exception during content extraction."""
    file_name = "test.pdf"
    model_name = "intfloat/multilingual-e5-large-instruct"
    chunk_size = 500
    chunk_overlap = 50
    file_reader = "simple"
    with patch.object(llmeowsschunks, 'get_simple_content_from_file', side_effect=RuntimeError("Extraction failed")):
        with pytest.raises(RuntimeError, match="Extraction failed"):
            llmeowsschunks.get_chunking(
                binary_file, file_name, model_name, chunk_size, chunk_overlap, file_reader
            )
        llmeowsschunks.logger.exception.assert_called_once_with(
            f"Error during chunking process for file `{file_name}`: Extraction failed"
        )

def test_get_chunking_invalid_chunk_size(llmeowsschunks, binary_file):
    """Test handling of invalid chunk_size."""
    file_name = "test.pdf"
    model_name = "intfloat/multilingual-e5-large-instruct"
    chunk_size = -1
    chunk_overlap = 50
    file_reader = "simple"
    with patch.object(llmeowsschunks, 'get_simple_content_from_file', return_value="Sample text"):
        with patch.object(llmeowsschunks, 'get_splitter') as mock_splitter:
            with pytest.raises(ValueError):
                llmeowsschunks.get_chunking(
                    binary_file, file_name, model_name, chunk_size, chunk_overlap, file_reader
                )
            mock_splitter.assert_not_called()

def test_get_chunking_empty_content(llmeowsschunks, binary_file):
    """Test chunking with empty file content."""
    file_name = "test.pdf"
    model_name = "intfloat/multilingual-e5-large-instruct"
    chunk_size = 500
    chunk_overlap = 50
    file_reader = "simple"
    mock_splitter = MagicMock()
    mock_splitter.split_text.return_value = []
    with patch.object(llmeowsschunks, 'get_simple_content_from_file', return_value=""):
        with patch.object(llmeowsschunks, 'get_splitter', return_value=mock_splitter):
            chunks = llmeowsschunks.get_chunking(
                binary_file, file_name, model_name, chunk_size, chunk_overlap, file_reader
            )
    assert chunks == []
    mock_splitter.split_text.assert_called_once_with("")

@pytest.mark.parametrize("reader_type, content_method", [
    ("simple", "get_simple_content_from_file"),
    ("ocr", "get_ocr_content_from_file"),
    ("markdown", "get_markdown_content_from_file"),
])
def test_get_chunking_readers(llmeowsschunks, binary_file, reader_type, content_method):
    """Test successful chunking for all reader types."""
    file_name = "test.pdf"
    model_name = "intfloat/multilingual-e5-large-instruct"
    chunk_size = 500
    chunk_overlap = 50
    mock_content = "Sample text content"
    mock_splitter = MagicMock()
    mock_chunks = ["chunk1", "chunk2"]
    mock_splitter.split_text.return_value = mock_chunks
    with patch.object(llmeowsschunks, content_method, return_value=mock_content) as mock_get_content:
        with patch.object(llmeowsschunks, 'get_splitter', return_value=mock_splitter) as mock_get_splitter:
            chunks = llmeowsschunks.get_chunking(
                binary_file, file_name, model_name, chunk_size, chunk_overlap, reader_type
            )
    assert chunks == mock_chunks
    llmeowsschunks.logger.debug.assert_any_call(
        f"Starting chunking process for file `{file_name}`, reader `{reader_type}`, model `{model_name}`."
    )
    llmeowsschunks.logger.info.assert_any_call(f"Using {reader_type} reader for file `{file_name}`")
    llmeowsschunks.logger.info.assert_any_call(
        f"Successfully split file `{file_name}` into {len(mock_chunks)} using model `{model_name}`"
    )
    mock_get_content.assert_called_once_with(binary_file, file_name)
    mock_get_splitter.assert_called_once_with(model_name, chunk_size, chunk_overlap)
    mock_splitter.split_text.assert_called_once_with(mock_content)
    llmeowsschunks.logger.exception.assert_not_called()

def test_get_chunking_unsupported_reader(llmeowsschunks, binary_file):
    """Test handling of unsupported file reader option."""
    file_name = "test.pdf"
    model_name = "intfloat/multilingual-e5-large-instruct"
    chunk_size = 500
    chunk_overlap = 50
    file_reader = "invalid"
    with patch.object(llmeowsschunks, 'get_simple_content_from_file') as mock_simple, \
         patch.object(llmeowsschunks, 'get_ocr_content_from_file') as mock_ocr, \
         patch.object(llmeowsschunks, 'get_markdown_content_from_file') as mock_md, \
         patch.object(llmeowsschunks, 'get_splitter') as mock_splitter:
        with pytest.raises(ValueError, match="Unsupported file reader: invalid"):
            llmeowsschunks.get_chunking(
                binary_file, file_name, model_name, chunk_size, chunk_overlap, file_reader
            )
        llmeowsschunks.logger.debug.assert_any_call(
            f"Starting chunking process for file `{file_name}`, reader `{file_reader}`, model `{model_name}`."
        )
        llmeowsschunks.logger.error.assert_called_once_with(f"Unsupported file reader option: {file_reader}")
        llmeowsschunks.logger.exception.assert_called_once_with(
            f"Error during chunking process for file `{file_name}`: Unsupported file reader: {file_reader}"
        )
        mock_simple.assert_not_called()
        mock_ocr.assert_not_called()
        mock_md.assert_not_called()
        mock_splitter.assert_not_called()
