import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from pathlib import Path
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

def run_unsupported_extension_test(llmeowsschunks, binary_file, file_name, method_name, error_msg, cleanup_msg):
    """Helper function to test handling of unsupported file extension for both content extraction and OCR."""
    file_path = str(Path(llmeowsschunks.TMP_DIR) / file_name)
    with patch.object(llmeowsschunks, '_save_temp_file', return_value=file_path):
        with patch.object(llmeowsschunks, '_cleanup_file') as mock_cleanup:
            with pytest.raises(ValueError, match=error_msg):
                getattr(llmeowsschunks, method_name)(binary_file, file_name)
    llmeowsschunks.logger.error.assert_called_once_with(error_msg)
    llmeowsschunks.logger.debug.assert_any_call(f"{cleanup_msg} `{file_path}`")
    mock_cleanup.assert_called_once_with(file_path)

@pytest.mark.parametrize("file_name", ["test.unsupported"])
def test_get_simple_content_unsupported_extension(llmeowsschunks, binary_file, file_name):
    """Test handling of unsupported file extension for content extraction."""
    run_unsupported_extension_test(
        llmeowsschunks, 
        binary_file, 
        file_name, 
        "get_simple_content_from_file", 
        "Unsupported file extension: .unsupported", 
        "Cleaning up temporary file"
    )

@pytest.mark.parametrize("file_name", ["test.unsupported"])
def test_get_ocr_content_unsupported_extension(llmeowsschunks, binary_file, file_name):
    """Test handling of unsupported file extension for OCR."""
    run_unsupported_extension_test(
        llmeowsschunks, 
        binary_file, 
        file_name, 
        "get_ocr_content_from_file", 
        "Unsupported file extension: .unsupported", 
        "Cleaning up uploaded file"
    )

@pytest.mark.parametrize("file_name", ["test.unsupported"])
def test_get_markdown_content_unsupported_extension(llmeowsschunks, binary_file, file_name):
    """Test handling of unsupported file extension for Markdown."""
    run_unsupported_extension_test(
        llmeowsschunks,
        binary_file,
        file_name,
        "get_markdown_content_from_file",
        "Unsupported file extension: .unsupported", 
        "Cleaning up uploaded file"
    )

def run_error_processing_test(method_name, error_msg_prefix, llmeowsschunks, binary_file, file_name, patch_target):
    file_path = str(Path(llmeowsschunks.TMP_DIR) / file_name)
    with patch.object(llmeowsschunks, '_save_temp_file', return_value=file_path):
        with patch.object(llmeowsschunks, '_cleanup_file') as mock_cleanup:
            with patch(patch_target, side_effect=Exception("Simulated error")):
                with pytest.raises(Exception, match="Simulated error"):
                    getattr(llmeowsschunks, method_name)(binary_file, file_name)
    getattr(llmeowsschunks.logger, "exception").assert_called_once_with(
        f"{error_msg_prefix} `{file_name}`: Simulated error"
    )
    llmeowsschunks.logger.debug.assert_any_call(
        f"Cleaning up {'temporary' if method_name == 'get_simple_content_from_file' else 'uploaded'} file `{file_path}`"
    )
    mock_cleanup.assert_called_once_with(file_path)

@pytest.mark.parametrize("method_name, file_name, patch_target, error_prefix", [
    ("get_simple_content_from_file", "test.pdf", "lolimeowss.main.pymupdf.open", "Error processing file"),
    ("get_simple_content_from_file", "test.txt", "lolimeowss.main.pymupdf.open", "Error processing file"),
    ("get_simple_content_from_file", "test.jpg", "lolimeowss.main.pymupdf.open", "Error processing file"),
    ("get_simple_content_from_file", "test.jpeg", "lolimeowss.main.pymupdf.open", "Error processing file"),
    ("get_simple_content_from_file", "test.png", "lolimeowss.main.pymupdf.open", "Error processing file"),
    ("get_simple_content_from_file", "test.py", "lolimeowss.main.pymupdf.open", "Error processing file"),
    ("get_simple_content_from_file", "test.xml", "lolimeowss.main.pymupdf.open", "Error processing file"),
    ("get_simple_content_from_file", "test.json", "lolimeowss.main.pymupdf.open", "Error processing file"),
    ("get_simple_content_from_file", "test.csv", "lolimeowss.main.pd.read_csv", "Error processing file"),
    ("get_simple_content_from_file", "test.xls", "lolimeowss.main.pd.read_excel", "Error processing file"),
    ("get_simple_content_from_file", "test.xlsx", "lolimeowss.main.pd.read_excel", "Error processing file"),
    ("get_ocr_content_from_file", "test.pdf", "lolimeowss.main.pymupdf.open", "Error during OCR extraction from file"),
    ("get_markdown_content_from_file", "test.pdf", "lolimeowss.main.DocumentConverter", "Error during Markdown convert extraction from file"),
])
def test_processing_error_by_type(llmeowsschunks, binary_file, method_name, file_name, patch_target, error_prefix):
    """Test for error propagation in file processing/OCR/Markdown."""
    run_error_processing_test(method_name, error_prefix, llmeowsschunks, binary_file, file_name, patch_target)

@pytest.mark.parametrize("file_name", ["test.unsupported"])
def test_get_markdown_content_unsupported_extension(llmeowsschunks, binary_file, file_name):
    """Test handling of unsupported file extension for Markdown."""
    run_unsupported_extension_test(
        llmeowsschunks, 
        binary_file, 
        file_name, 
        "get_markdown_content_from_file", 
        "Unsupported file extension: .unsupported", 
        "Cleaning up uploaded file"
    )

def run_get_simple_content_test(llmeowsschunks, binary_file, file_name, filetype_arg=None):
    """Helper function to test content extraction from files supported by PyMuPDF."""
    file_path = str(Path(llmeowsschunks.TMP_DIR) / file_name)
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "page content"
    mock_doc.__iter__.return_value = [mock_page]
    mock_doc.__enter__.return_value = mock_doc
    with patch.object(llmeowsschunks, '_save_temp_file', return_value=file_path):
        with patch.object(llmeowsschunks, '_cleanup_file') as mock_cleanup:
            with patch('lolimeowss.main.pymupdf.open', return_value=mock_doc) as mock_pymupdf:
                content = llmeowsschunks.get_simple_content_from_file(binary_file, file_name)
    assert content == "page content"
    llmeowsschunks.logger.debug.assert_any_call(f"Starting content extraction from `{file_name}`")
    llmeowsschunks.logger.debug.assert_any_call(f"File saved to `{file_path}` with extension: .{file_name.split('.')[-1]}")
    llmeowsschunks.logger.debug.assert_any_call("Detected file type supported by PyMuPDF. Reading with PyMuPDF.")
    llmeowsschunks.logger.info.assert_called_once_with(f"Successfully extracted content using PyMuPDF: {file_name}")
    if filetype_arg:
        mock_pymupdf.assert_called_once_with(file_path, filetype=filetype_arg)
    else:
        mock_pymupdf.assert_called_once_with(file_path)
    mock_cleanup.assert_called_once_with(file_path)

@pytest.mark.parametrize("file_name", ["test.pdf", "test.txt", "test.jpg", "test.jpeg", "test.png"])
def test_get_simple_content_text(llmeowsschunks, binary_file, file_name):
    """Test content extraction from files supported by PyMuPDF."""
    run_get_simple_content_test(llmeowsschunks, binary_file, file_name)

@pytest.mark.parametrize("file_name", ["test.py", "test.xml", "test.json"])
def test_get_simple_content_as_text(llmeowsschunks, binary_file, file_name):
    """Test content extraction from files supported by PyMuPDF."""
    run_get_simple_content_test(llmeowsschunks, binary_file, file_name, filetype_arg="txt")

def test_get_simple_content_csv(llmeowsschunks, binary_file):
    """Test content extraction from a CSV file."""
    file_name = "test.csv"
    mock_df = MagicMock()
    mock_df.to_string.return_value = "col1,col2\nval1,val2"
    file_path = str(Path(llmeowsschunks.TMP_DIR) / file_name)
    with patch.object(llmeowsschunks, '_save_temp_file', return_value=file_path):
        with patch.object(llmeowsschunks, '_cleanup_file') as mock_cleanup:
            with patch('lolimeowss.main.pd.read_csv', return_value=mock_df) as mock_read_csv:
                content = llmeowsschunks.get_simple_content_from_file(binary_file, file_name)
    assert content == "col1,col2\nval1,val2"
    llmeowsschunks.logger.debug.assert_any_call(f"Starting content extraction from `{file_name}`")
    llmeowsschunks.logger.debug.assert_any_call(f"File saved to `{file_path}` with extension: .csv")
    llmeowsschunks.logger.debug.assert_any_call("Detected CSV file. Reading with pandas.")
    llmeowsschunks.logger.info.assert_called_once_with(f"Successfully extracted content from CSV: {file_name}")
    mock_read_csv.assert_called_once_with(file_path)
    mock_cleanup.assert_called_once_with(file_path)

@pytest.mark.parametrize("file_name", ["test.xlsx", "test.xls"])
def test_get_simple_content_excel(llmeowsschunks, binary_file, file_name):
    """Test content extraction from Excel files (.xlsx and .xls)."""
    mock_df = MagicMock()
    mock_df.to_string.return_value = "col1,col2\nval1,val2"
    file_path = str(Path(llmeowsschunks.TMP_DIR) / file_name)
    with patch.object(llmeowsschunks, '_save_temp_file', return_value=file_path):
        with patch.object(llmeowsschunks, '_cleanup_file') as mock_cleanup:
            with patch('lolimeowss.main.pd.read_excel', return_value=mock_df) as mock_read_excel:
                content = llmeowsschunks.get_simple_content_from_file(binary_file, file_name)
    assert content == "col1,col2\nval1,val2"
    llmeowsschunks.logger.debug.assert_any_call(f"Starting content extraction from `{file_name}`")
    llmeowsschunks.logger.debug.assert_any_call(f"File saved to `{file_path}` with extension: .{file_name.split('.')[-1]}")
    llmeowsschunks.logger.debug.assert_any_call("Detected Excel file. Reading with pandas.")
    llmeowsschunks.logger.info.assert_called_once_with(f"Successfully extracted content from Excel: {file_name}")
    mock_read_excel.assert_called_once_with(file_path)
    mock_cleanup.assert_called_once_with(file_path)

@pytest.mark.parametrize("file_name", ["test.pdf"])
def test_get_ocr_content(llmeowsschunks, binary_file, file_name):
    """Test OCR extraction from files."""
    temp_file_path = str(Path(llmeowsschunks.TMP_DIR) / file_name)
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.number = 0
    mock_pixmap = MagicMock()
    mock_page.get_pixmap.return_value = mock_pixmap
    mock_doc.__iter__.return_value = [mock_page]
    mock_doc.__enter__.return_value = mock_doc
    mock_image = MagicMock()
    mock_image_path = "page-0.png"
    with patch.object(llmeowsschunks, '_save_temp_file', return_value=temp_file_path):
        with patch.object(llmeowsschunks, '_cleanup_file') as mock_cleanup:
            with patch('lolimeowss.main.pymupdf.open', return_value=mock_doc) as mock_pymupdf:
                with patch('lolimeowss.main.Image.open', return_value=mock_image) as mock_image_open:
                    with patch('lolimeowss.main.pytesseract.image_to_string', return_value="extracted text") as mock_ocr:
                        with patch('lolimeowss.main.os.path.exists', return_value=True):
                            with patch('lolimeowss.main.os.remove') as mock_remove:
                                content = llmeowsschunks.get_ocr_content_from_file(binary_file, file_name)
    assert content == "extracted text\n"
    llmeowsschunks.logger.debug.assert_any_call(f"Starting OCR extraction from `{file_name}`")
    llmeowsschunks.logger.debug.assert_any_call(f"File saved to `{temp_file_path}` with extension: .pdf")
    llmeowsschunks.logger.debug.assert_any_call("Detected PDF file. Converting pages to images for OCR.")
    llmeowsschunks.logger.debug.assert_any_call(f"Saved page `0` as image: {mock_image_path}")
    llmeowsschunks.logger.info.assert_any_call(f"OCR extracted from page `0` of `{file_name}`")
    llmeowsschunks.logger.info.assert_any_call(f"Successfully OCR extracted from PDF: {file_name}")
    llmeowsschunks.logger.debug.assert_any_call(f"Deleted temporary image `{mock_image_path}`")
    llmeowsschunks.logger.debug.assert_any_call(f"Cleaning up uploaded file `{temp_file_path}`")
    mock_pymupdf.assert_called_once_with(temp_file_path)
    mock_image_open.assert_called_once_with(mock_image_path)
    mock_ocr.assert_called_once_with(mock_image, lang="tha+eng")
    mock_remove.assert_called_once_with(mock_image_path)
    mock_cleanup.assert_called_once_with(temp_file_path)

@pytest.mark.parametrize("file_name", ["test.pdf", "test.docx", "test.xlsx", "test.html", "test.pptx"])
def test_get_markdown_content(llmeowsschunks, binary_file, file_name):
    """Test markdown extraction from files."""
    file_path = str(Path(llmeowsschunks.TMP_DIR) / file_name)
    file_ext = Path(file_name).suffix.lower()
    mock_converter = MagicMock()
    mock_doc = MagicMock()
    mock_doc.document.export_to_markdown.return_value = "# Extracted Content\nSome text"
    mock_converter.convert.return_value = mock_doc
    with patch.object(llmeowsschunks, '_save_temp_file', return_value=file_path):
        with patch.object(llmeowsschunks, '_cleanup_file') as mock_cleanup:
            with patch('lolimeowss.main.DocumentConverter', return_value=mock_converter):
                content = llmeowsschunks.get_markdown_content_from_file(binary_file, file_name)
    assert content == "# Extracted Content\nSome text"
    llmeowsschunks.logger.debug.assert_any_call(f"Starting Markdown convert extraction from `{file_name}`")
    llmeowsschunks.logger.debug.assert_any_call(f"File saved to `{file_path}` with extension: {file_ext}")
    llmeowsschunks.logger.debug.assert_any_call(f"Detected `{file_ext}` file. Converting to Markdown using Docling.")
    llmeowsschunks.logger.info.assert_called_once_with(f"Successfully Markdown extracted from {file_ext}: {file_name}")
    llmeowsschunks.logger.debug.assert_any_call(f"Cleaning up uploaded file `{file_path}`")
    mock_converter.convert.assert_called_once_with(file_path)
    mock_doc.document.export_to_markdown.assert_called_once()
    mock_cleanup.assert_called_once_with(file_path)
