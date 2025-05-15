import pytest
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock
from lolimeowss.main import LLMeowsSChunks

@pytest.fixture
def llmeowsschunks(tmp_path):
    """Fixture to initialize LLMeowsSChunks with a temporary directory and mocked logger."""
    llms_chunks = LLMeowsSChunks()
    llms_chunks.TMP_DIR = str(tmp_path) + "/"
    llms_chunks.logger = MagicMock()
    return llms_chunks

def test_save_temp_file_success(llmeowsschunks):
    """Test _save_temp_file saves a file correctly and returns its path."""
    file_content = b"Meows, LLMeowsS~"
    file = BytesIO(file_content)
    file_name = "test_llmeowss.txt"
    file_path = llmeowsschunks._save_temp_file(file, file_name)
    expected_path = os.path.join(llmeowsschunks.TMP_DIR, file_name)
    assert file_path == expected_path
    with open(file_path, "rb") as f:
        assert f.read() == file_content
    llmeowsschunks.logger.debug.assert_called_with(f"Preparing to save file `{file_name}` to `{file_path}`")
    llmeowsschunks.logger.info.assert_called_with(f"Succesfully saved file to: {file_path}")

def test_save_temp_file_io_error(llmeowsschunks, monkeypatch):
    """Test _save_temp_file handles IO errors."""
    file_content = b"Meows, LLMeowsS~"
    file = BytesIO(file_content)
    file_name = "test_llmeowss.txt"
    def mock_open(*args, **kwargs):
        raise IOError("Permission denied")
    monkeypatch.setattr("builtins.open", mock_open)
    file_path = llmeowsschunks._save_temp_file(file, file_name)
    expected_path = os.path.join(llmeowsschunks.TMP_DIR, file_name)
    assert not os.path.exists(expected_path)
    assert file_path == expected_path
    llmeowsschunks.logger.debug.assert_called_with(f"Preparing to save file `{file_name}` to `{expected_path}`")
    llmeowsschunks.logger.exception.assert_called_with(f"Failed to save file `{file_name}` to `{expected_path}`: Permission denied")

def test_cleanup_file_exists(llmeowsschunks):
    """Test _cleanup_file deletes an existing file."""
    file_name = "test_llmeowss.txt"
    file_path = os.path.join(llmeowsschunks.TMP_DIR, file_name)
    with open(file_path, "wb") as f:
        f.write(b"Meows, LLMeows~")
    llmeowsschunks._cleanup_file(file_path)
    assert not os.path.exists(file_path)
    llmeowsschunks.logger.debug.assert_called_with(f"Attempting to delete `{file_path}`")
    llmeowsschunks.logger.info.assert_called_with(f"Successfully deleted file: {file_path}")

def test_cleanup_file_nonexistent(llmeowsschunks):
    """Test _cleanup_file handles nonexistent files."""
    file_name = "nonexistent.txt"
    file_path = os.path.join(llmeowsschunks.TMP_DIR, file_name)
    llmeowsschunks._cleanup_file(file_path)
    assert not os.path.exists(file_path)
    llmeowsschunks.logger.debug.assert_called_with(f"Attempting to delete `{file_path}`")
    llmeowsschunks.logger.warning.assert_called_with(f"Tried to delete file, but it does not exist: {file_path}")

def test_cleanup_file_os_error(llmeowsschunks, monkeypatch):
    """Test _cleanup_file handles OS errors during deletion."""
    file_name = "test_llmeowss.txt"
    file_path = os.path.join(llmeowsschunks.TMP_DIR, file_name)
    with open(file_path, "wb") as f:
        f.write(b"Meows, LLMeows~")
    def mock_remove(*args, **kwargs):
        raise OSError("Permission denied")
    monkeypatch.setattr(os, "remove", mock_remove)
    llmeowsschunks._cleanup_file(file_path)
    assert os.path.exists(file_path)
    llmeowsschunks.logger.debug.assert_called_with(f"Attempting to delete `{file_path}`")
    llmeowsschunks.logger.exception.assert_called_with(f"Failed to delete file `{file_path}`: Permission denied")