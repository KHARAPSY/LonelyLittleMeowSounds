import pytest
from unittest.mock import patch, MagicMock
from lolimeowss.main import LLMeowsSChunks

@pytest.fixture
def llmeowsschunks(tmp_path):
    """Fixture to initialize LLMeowsSChunks with a temporary directory and mocked logger."""
    llms_chunks = LLMeowsSChunks()
    llms_chunks.TMP_DIR = str(tmp_path) + "/"
    llms_chunks.logger = MagicMock()
    return llms_chunks

def test_get_tokenizer_preloaded(llmeowsschunks):
    """Test retrieving a tokenizer from preloaded_tokenizers."""
    model_name = "intfloat/multilingual-e5-large-instruct"
    mock_tokenizer = MagicMock()
    llmeowsschunks.preloaded_tokenizers[model_name] = mock_tokenizer
    tokenizer = llmeowsschunks.get_tokenizer(model_name)
    assert tokenizer == mock_tokenizer
    llmeowsschunks.logger.debug.assert_any_call(f"Request received for tokenizer: {model_name}")
    llmeowsschunks.logger.debug.assert_any_call(f"Tokenizer for `{model_name}` found in preloaded tokenizers.")
    llmeowsschunks.logger.info.assert_not_called()
    assert model_name not in llmeowsschunks.tokenizer_cache

@patch('lolimeowss.main.AutoTokenizer')
def test_get_tokenizer_cached(mock_autotokenizer, llmeowsschunks):
    """Test retrieving a tokenizer from tokenizer_cache."""
    model_name = "new-model"
    mock_tokenizer = MagicMock()
    llmeowsschunks.tokenizer_cache[model_name] = mock_tokenizer
    tokenizer = llmeowsschunks.get_tokenizer(model_name)
    assert tokenizer == mock_tokenizer
    llmeowsschunks.logger.debug.assert_any_call(f"Request received for tokenizer: {model_name}")
    mock_autotokenizer.from_pretrained.assert_not_called()
    assert model_name in llmeowsschunks.tokenizer_cache

@patch('lolimeowss.main.AutoTokenizer')
def test_get_tokenizer_load_and_cache(mock_autotokenizer, llmeowsschunks):
    """Test loading and caching a new tokenizer from Hugging Face."""
    model_name = "new-model"
    mock_tokenizer = MagicMock()
    mock_autotokenizer.from_pretrained.return_value = mock_tokenizer
    tokenizer = llmeowsschunks.get_tokenizer(model_name)
    assert tokenizer == mock_tokenizer
    llmeowsschunks.logger.debug.assert_any_call(f"Request received for tokenizer: {model_name}")
    llmeowsschunks.logger.info.assert_any_call(f"Tokenizer for `{model_name}` not cached. Loading from Hugging Face.")
    llmeowsschunks.logger.info.assert_any_call(f"Successfully loaded and cached tokenizer for `{model_name}`")
    mock_autotokenizer.from_pretrained.assert_called_once_with(model_name)
    assert llmeowsschunks.tokenizer_cache[model_name] == mock_tokenizer

@patch('lolimeowss.main.AutoTokenizer')
def test_get_tokenizer_load_failure(mock_autotokenizer, llmeowsschunks):
    """Test handling of error when loading tokenizer fails."""
    model_name = "invalid-model"
    mock_autotokenizer.from_pretrained.side_effect = Exception("Load error")
    with pytest.raises(Exception, match="Load error"):
        llmeowsschunks.get_tokenizer(model_name)
    llmeowsschunks.logger.debug.assert_any_call(f"Request received for tokenizer: {model_name}")
    llmeowsschunks.logger.info.assert_any_call(f"Tokenizer for `{model_name}` not cached. Loading from Hugging Face.")
    llmeowsschunks.logger.exception.assert_called_once_with(f"Failed to load tokenizer for `{model_name}`: Load error")
    assert model_name not in llmeowsschunks.tokenizer_cache

@pytest.mark.parametrize("model_name", ["text-embedding-ada-002"])
def test_get_splitter_tiktoken_model(llmeowsschunks, model_name):
    """Test creating a TokenTextSplitter for text-embedding-ada-002 using tiktoken."""
    chunk_size = 1000
    chunk_overlap = 50
    mock_splitter = MagicMock()
    with patch('lolimeowss.main.TokenTextSplitter') as mock_token_splitter:
        mock_token_splitter.from_tiktoken_encoder.return_value = mock_splitter
        splitter = llmeowsschunks.get_splitter(model_name, chunk_size, chunk_overlap)
    assert splitter == mock_splitter
    llmeowsschunks.logger.debug.assert_any_call(f"Creating TokenTextSplitter for model `{model_name}`")
    llmeowsschunks.logger.info.assert_called_once_with(f"Using tiktoken encoder for model `{model_name}`")
    mock_token_splitter.from_tiktoken_encoder.assert_called_once_with(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    llmeowsschunks.logger.exception.assert_not_called()

@pytest.mark.parametrize("model_name", [
    "intfloat/multilingual-e5-large-instruct",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "Lajavaness/bilingual-embedding-large",
    "BAAI/bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
])
def test_get_splitter_huggingface_model(llmeowsschunks, model_name):
    """Test creating a TokenTextSplitter for a Hugging Face model."""
    chunk_size = 800
    chunk_overlap = 100
    mock_tokenizer = MagicMock()
    mock_splitter = MagicMock()
    with patch.object(llmeowsschunks, 'get_tokenizer', return_value=mock_tokenizer) as mock_get_tokenizer:
        with patch('lolimeowss.main.TokenTextSplitter') as mock_token_splitter:
            mock_token_splitter.from_huggingface_tokenizer.return_value = mock_splitter
            splitter = llmeowsschunks.get_splitter(model_name, chunk_size, chunk_overlap)
    assert splitter == mock_splitter
    llmeowsschunks.logger.debug.assert_any_call(f"Creating TokenTextSplitter for model `{model_name}`")
    llmeowsschunks.logger.info.assert_called_once_with(f"Using Hugging Face tokenizer for model `{model_name}`")
    mock_get_tokenizer.assert_called_once_with(model_name)
    mock_token_splitter.from_huggingface_tokenizer.assert_called_once_with(
        mock_tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    llmeowsschunks.logger.exception.assert_not_called()

@pytest.mark.parametrize("model_name", [
    "intfloat/multilingual-e5-large-instruct",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "Lajavaness/bilingual-embedding-large",
    "BAAI/bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
])
def test_get_splitter_default_chunk_params(llmeowsschunks, model_name):
    """Test default chunk_size and chunk_overlap values."""
    chunk_size = 0
    chunk_overlap = 0
    mock_tokenizer = MagicMock()
    mock_splitter = MagicMock()
    with patch.object(llmeowsschunks, 'get_tokenizer', return_value=mock_tokenizer):
        with patch('lolimeowss.main.TokenTextSplitter') as mock_token_splitter:
            mock_token_splitter.from_huggingface_tokenizer.return_value = mock_splitter
            splitter = llmeowsschunks.get_splitter(model_name, chunk_size, chunk_overlap)
    assert splitter == mock_splitter
    llmeowsschunks.logger.debug.assert_any_call(f"Creating TokenTextSplitter for model `{model_name}`")
    llmeowsschunks.logger.debug.assert_any_call("chunk_size set to default: 500")
    llmeowsschunks.logger.debug.assert_any_call("chunk_overlap set to default: 0")
    llmeowsschunks.logger.info.assert_called_once_with(f"Using Hugging Face tokenizer for model `{model_name}`")
    mock_token_splitter.from_huggingface_tokenizer.assert_called_once_with(
        mock_tokenizer,
        chunk_size=500,
        chunk_overlap=0
    )
    llmeowsschunks.logger.exception.assert_not_called()

@patch('lolimeowss.main.TokenTextSplitter')
@pytest.mark.parametrize("model_name", [
    "intfloat/multilingual-e5-large-instruct",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "Lajavaness/bilingual-embedding-large",
    "BAAI/bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
])
def test_get_splitter_creation_failure(mock_token_splitter_class, llmeowsschunks, model_name):
    """Test handling of error when creating TokenTextSplitter fails."""
    chunk_size = 800
    chunk_overlap = 100
    mock_tokenizer = MagicMock()
    mock_token_splitter_class.from_huggingface_tokenizer.side_effect = Exception("Splitter error")
    with patch.object(llmeowsschunks, 'get_tokenizer', return_value=mock_tokenizer):
        with pytest.raises(Exception, match="Splitter error"):
            llmeowsschunks.get_splitter(model_name, chunk_size, chunk_overlap)
    llmeowsschunks.logger.debug.assert_any_call(f"Creating TokenTextSplitter for model `{model_name}`")
    llmeowsschunks.logger.info.assert_called_once_with(f"Using Hugging Face tokenizer for model `{model_name}`")
    llmeowsschunks.logger.exception.assert_called_once_with(
        f"Failed to create TokenTextSplitter for model `{model_name}`: Splitter error"
    )
    mock_token_splitter_class.from_huggingface_tokenizer.assert_called_once_with(
        mock_tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )