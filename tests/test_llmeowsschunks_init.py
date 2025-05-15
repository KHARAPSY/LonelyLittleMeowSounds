import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from lolimeowss.main import LLMeowsSChunks

@pytest.fixture
def llmeowsschunks(tmp_path):
    mock_logger = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.exception = MagicMock()

    mock_tokenizer = MagicMock()

    with patch('lolimeowss.main.setup_logger', return_value=mock_logger), \
         patch('lolimeowss.main.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('os.makedirs') as mock_makedirs:
        
        llms_chunks = LLMeowsSChunks()
        # Override TMP_DIR with a test directory after init
        llms_chunks.TMP_DIR = str(tmp_path) + "/"
        llms_chunks.logger = mock_logger
        return llms_chunks

def test_init_successful(llmeowsschunks):
    """Test successful initialization of LLMeowsSChunks using the fixture."""
    expected_model_ids = [
        "intfloat/multilingual-e5-large-instruct",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "Lajavaness/bilingual-embedding-large",
        "BAAI/bge-m3",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
    ]

    assert llmeowsschunks.TMP_DIR.endswith('/')
    assert isinstance(llmeowsschunks.logger, MagicMock)
    assert llmeowsschunks.preloaded_model_ids == expected_model_ids
    assert llmeowsschunks.tokenizer_cache == {}
    assert set(llmeowsschunks.preloaded_tokenizers.keys()) == set(expected_model_ids)

    # Verify logger calls
    llmeowsschunks.logger.debug.assert_any_call("Tokenizer cache initialized as empty.")
    for model_id in expected_model_ids:
        llmeowsschunks.logger.debug.assert_any_call(f"Loading tokenizer for model `{model_id}`")
        llmeowsschunks.logger.info.assert_any_call(f"Successfully loaded tokenizer for model: {model_id}")

def test_init_with_tokenizer_failure(tmp_path):
    """Test LLMeowsSChunks initialization when tokenizer loading fails."""
    mock_logger = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.exception = MagicMock()

    def tokenizer_side_effect(model_id):
        raise RuntimeError(f"Failed to load tokenizer for {model_id}")

    with patch('lolimeowss.main.setup_logger', return_value=mock_logger), \
         patch('lolimeowss.main.AutoTokenizer.from_pretrained', side_effect=tokenizer_side_effect), \
         patch('os.makedirs'):

        llms_chunks = LLMeowsSChunks()
        llms_chunks.TMP_DIR = str(tmp_path) + "/"
        llms_chunks.logger = mock_logger

        assert llms_chunks.preloaded_tokenizers == {}
        assert llms_chunks.tokenizer_cache == {}

        # Check that exception logging occurred for each model
        for model_id in llms_chunks.preloaded_model_ids:
            mock_logger.debug.assert_any_call(f"Loading tokenizer for model `{model_id}`")
            mock_logger.exception.assert_any_call(
                f"Failed to load tokenizer for `{model_id}`: Failed to load tokenizer for {model_id}"
            )
