import os
import pymupdf
import pytesseract
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import BinaryIO
from langchain_text_splitters import TokenTextSplitter
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lolimeowss.logging_llmeowss import setup_logger

class LLMeowsSChunks:

    def __init__(self):
        """
        Initializes the LLMeowsSChunks class.

        This sets up temporary directories and preloads tokenizers for a selection of commonly used Hugging Face models. Tokenizers for other models are loaded on demand and cached for reuse.
        """
        self.TMP_DIR = 'tmp_file/'
        self.logger = setup_logger(__name__)
        
        self.logger.info("Initializing LLMeowsSChunks")
        try:
            os.makedirs(self.TMP_DIR, exist_ok=True)
            self.logger.debug(f"Temporary directory ensured at: {self.TMP_DIR}")
        except Exception as e:
            self.logger.exception(f"Failed to create temp directory `{self.TMP_DIR}`: {e}")
        self.preloaded_model_ids = [
            "intfloat/multilingual-e5-large-instruct",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "Lajavaness/bilingual-embedding-large",
            "BAAI/bge-m3",
            "Snowflake/snowflake-arctic-embed-l-v2.0",
        ]
        self.preloaded_tokenizers = {}
        for model_id in self.preloaded_model_ids:
            try:
                self.logger.debug(f"Loading tokenizer for model `{model_id}`")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.preloaded_tokenizers[model_id] = tokenizer
                self.logger.info(f"Successfully loaded tokenizer for model: {model_id}")
            except Exception as e:
                self.logger.exception(f"Failed to load tokenizer for `{model_id}`: {e}")
        self.tokenizer_cache = {}
        self.logger.debug("Tokenizer cache initialized as empty.")

    def hi_llmeowss(self):
        self.logger.info("Say hi to llmeowss!")
        print("Meowss, llmeowss is you pal-ai ready to help~")

    def _save_temp_file(self, file: BinaryIO, file_name: str) -> str:
        """
        Saves an uploaded file to a temporary directory.

        Args:
            file (BinaryIO): A file-like object.
            file_name (str): Name of the file to be saved.

        Returns:
            str: Full path to the saved file.
        """
        file_path = os.path.join(self.TMP_DIR, file_name)
        self.logger.debug(f"Preparing to save file `{file_name}` to `{file_path}`")
        try:
            with open(file_path, 'wb') as f:
                f.write(file.read())
            self.logger.info(f"Succesfully saved file to: {file_path}")
            return file_path
        except Exception as e:
            self.logger.exception(f"Failed to save file `{file_name}` to `{file_path}`: {e}")

    def _cleanup_file(self, file_path: str):
        """
        Deletes a file from the filesystem if it exists.

        Args:
            file_path (str): The path of the file to delete.
        """
        self.logger.debug(f"Attempting to delete `{file_path}`")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                self.logger.info(f"Successfully deleted file: {file_path}")
            except Exception as e:
                self.logger.exception(f"Failed to delete file `{file_path}`: {e}")
        else:
            self.logger.warning(f"Tried to delete file, but it does not exist: {file_path}")

    def get_simple_content_from_file(self, file: BinaryIO, file_name: str) -> str:
        """
        Extracts plain text content from supported file types.

        Args:
            file (BinaryIO): A file-like object to read from.
            file_name (str): The original name of the uploaded file.

        Returns:
            str: Extracted textual content.

        Raises:
            ValueError: If the file type is unsupported.
        """
        self.logger.debug(f"Starting content extraction from `{file_name}`")
        file_path = self._save_temp_file(file, file_name)
        file_ext = Path(file_path).suffix.lower()
        self.logger.debug(f"File saved to `{file_path}` with extension: {file_ext}")
        content = ""
        try:
            match file_ext:
                case ".pdf" | ".txt" | ".jpg" | ".jpeg" | ".png":
                    self.logger.debug("Detected file type supported by PyMuPDF. Reading with PyMuPDF.")
                    with pymupdf.open(file_path) as doc:
                        content = "".join(page.get_text() for page in doc)
                    self.logger.info(f"Successfully extracted content using PyMuPDF: {file_name}")
                case ".py" | ".xml" | ".json":
                    self.logger.debug("Detected file type supported by PyMuPDF. Reading with PyMuPDF.")
                    with pymupdf.open(file_path, filetype="txt") as doc:
                        content = "".join(page.get_text() for page in doc)
                    self.logger.info(f"Successfully extracted content using PyMuPDF: {file_name}")
                case ".csv":
                    self.logger.debug("Detected CSV file. Reading with pandas.")
                    df = pd.read_csv(file_path)
                    content = df.to_string(index=False)
                    self.logger.info(f"Successfully extracted content from CSV: {file_name}")
                case ".xls" | ".xlsx":
                    self.logger.debug("Detected Excel file. Reading with pandas.")
                    df = pd.read_excel(file_path)
                    content = df.to_string(index=False)
                    self.logger.info(f"Successfully extracted content from Excel: {file_name}")
                case _:
                    self.logger.error(f"Unsupported file extension: {file_ext}")
                    raise ValueError(f"Unsupported file extension: {file_ext}")
        except Exception as e:
            self.logger.exception(f"Error processing file `{file_name}`: {e}")
            raise
        finally:
            self.logger.debug(f"Cleaning up temporary file `{file_path}`")
            self._cleanup_file(file_path)
        return content
    
    def get_ocr_content_from_file(self, file: BinaryIO, file_name: str) -> str:
        """
        Uses OCR to extract text content from supported file types.

        Args:
            file (BinaryIO): A file-like object to read from.
            file_name (str): The original name of the uploaded file.

        Returns:
            str: Extracted textual content.
        
        Raises:
            ValueError: If the file type is unsupported.
        """
        self.logger.debug(f"Starting OCR extraction from `{file_name}`")
        file_path = self._save_temp_file(file, file_name)
        file_ext = Path(file_path).suffix.lower()
        self.logger.debug(f"File saved to `{file_path}` with extension: {file_ext}")
        content = ""
        try:
            match file_ext:
                case ".pdf":
                    self.logger.debug("Detected PDF file. Converting pages to images for OCR.")
                    doc = pymupdf.open(file_path)
                    for page in doc:
                        try:
                            pix = page.get_pixmap()
                            img_path = f"page-{page.number}.png"
                            pix.save(img_path)
                            self.logger.debug(f"Saved page `{page.number}` as image: {img_path}")

                            image = Image.open(img_path)
                            page_text = pytesseract.image_to_string(image, lang="tha+eng")
                            content += page_text + "\n"
                            self.logger.info(f"OCR extracted from page `{page.number}` of `{file_name}`")
                        except Exception as e:
                            self.logger.exception(f"Failed OCR on page `{page.number}` of `{file_name}`: {e}")
                        finally:
                            if os.path.exists(img_path):
                                os.remove(img_path)
                                self.logger.debug(f"Deleted temporary image `{img_path}`")
                    self.logger.info(f"Successfully OCR extracted from PDF: {file_name}")
                case _:
                    self.logger.error(f"Unsupported file extension: {file_ext}")
                    raise ValueError(f"Unsupported file extension: {file_ext}")
        except Exception as e:
            self.logger.exception(f"Error during OCR extraction from file `{file_name}`: {e}")
            raise
        finally:
            self.logger.debug(f"Cleaning up uploaded file `{file_path}`")
            self._cleanup_file(file_path)
        return content
    
    def get_markdown_content_from_file(self, file: BinaryIO, file_name: str) -> str:
        """
        Uses Docling to load text content from supported file types.

        Args:
            file (BinaryIO): A file-like object to read from.
            file_name (str): The original name of the uploaded file.

        Returns:
            str: Extracted textual content.
        
        Raises:
            ValueError: If the file type is unsupported.
        """
        self.logger.debug(f"Starting Markdown convert extraction from `{file_name}`")
        file_path = self._save_temp_file(file, file_name)
        file_ext = Path(file_path).suffix.lower()
        self.logger.debug(f"File saved to `{file_path}` with extension: {file_ext}")
        content = ""
        try:
            match file_ext:
                case ".pdf" | ".docx" | ".xlsx" | ".html" | ".pptx":
                    self.logger.debug(f"Detected `{file_ext}` file. Converting to Markdown using Docling.")
                    doc = DocumentConverter().convert(file_path)
                    content = doc.document.export_to_markdown()
                    self.logger.info(f"Successfully Markdown extracted from {file_ext}: {file_name}")
                case _:
                    self.logger.error(f"Unsupported file extension: {file_ext}")
                    raise ValueError(f"Unsupported file extension: {file_ext}")
        except Exception as e:
            self.logger.exception(f"Error during Markdown convert extraction from file `{file_name}`: {e}")
            raise
        finally:
            self.logger.debug(f"Cleaning up uploaded file `{file_path}`")
            self._cleanup_file(file_path)
        return content

    def get_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        """
        Retrieves a tokenizer for a specified Hugging Face model.

        Args:
            model_name (str): The model ID to retrieve the tokenizer for.

        Returns:
            PreTrainedTokenizerBase: Tokenizer instance.
        """
        self.logger.debug(f"Request received for tokenizer: {model_name}")
        if model_name in self.preloaded_tokenizers:
            self.logger.debug(f"Tokenizer for `{model_name}` found in preloaded tokenizers.")
            return self.preloaded_tokenizers[model_name]

        if model_name not in self.tokenizer_cache:
            try:
                self.logger.info(f"Tokenizer for `{model_name}` not cached. Loading from Hugging Face.")
                self.tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
                self.logger.info(f"Successfully loaded and cached tokenizer for `{model_name}`")
            except Exception as e:
                self.logger.exception(f"Failed to load tokenizer for `{model_name}`: {e}")
                raise
        return self.tokenizer_cache[model_name]

    def get_splitter(self, model_name: str, chunk_size: int, chunk_overlap: int) -> TokenTextSplitter:
        """
        Creates a TokenTextSplitter for the specified model and parameters.

        Args:
            model_name (str): The model ID for which the splitter should be configured.
            chunk_size (int): Maximum tokens per chunk. Defaults to 500 if 0.
            chunk_overlap (int): Maximum overlap between chunks. Defaults to 0 if 0.

        Returns:
            TokenTextSplitter: A text splitter configured for the model's tokenizer.
        """
        self.logger.debug(f"Creating TokenTextSplitter for model `{model_name}`")
        if chunk_size == 0:
            chunk_size = 500
            self.logger.debug(f"chunk_size set to default: {chunk_size}")
        if chunk_overlap == 0:
            chunk_overlap = 0
            self.logger.debug(f"chunk_overlap set to default: {chunk_overlap}")
        try:
            match model_name:
                case "text-embedding-ada-002":
                    self.logger.info(f"Using tiktoken encoder for model `{model_name}`")
                    return TokenTextSplitter.from_tiktoken_encoder(
                        model_name=model_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                case _:
                    self.logger.info(f"Using Hugging Face tokenizer for model `{model_name}`")
                    tokenizer = self.get_tokenizer(model_name)
                    return TokenTextSplitter.from_huggingface_tokenizer(
                        tokenizer,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
        except Exception as e:
            self.logger.exception(f"Failed to create TokenTextSplitter for model `{model_name}`: {e}")
            raise

    def get_chunking(self, file: BinaryIO, file_name: str, model_name: str, chunk_size: int, chunk_overlap: int, file_reader: str) -> list:
        """
        Extracts textual content from a file using the specified reading strategy and splits it into token-based chunks suitable for language model input.

        This method supports different content extraction strategies (plain text, OCR, or markdown-based) and uses a tokenizer-aware text splitter to segment the content into chunks with optional token overlap.

        Args:
            file (BinaryIO): File-like object to process.
            file_name (str): Name of the uploaded file.
            model_name (str): Embedding model for tokenizer/splitter.
            chunk_size (int): Maximum number of tokens per chunk.
            chunk_overlap (int): Overlap of tokens between chunks.
            file_reader (str): The method used to extract content. Must be one of:
                - "simple": For reading plain content.
                - "ocr": For extracting text from image-based.
                - "markdown": For reading content via Docling

        Returns:
            list[str]: A list of token-aware text chunks derived from the input file content.

        Raises:
            ValueError: If the file type or reading strategy is unsupported.
            Exception: If an error occurs during file reading or chunk splitting.
        """
        self.logger.debug(f"Starting chunking process for file `{file_name}`, reader `{file_reader}`, model `{model_name}`.")
        try:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be greater than 0.")
            if chunk_overlap < 0:
                raise ValueError("chunk_overlap must be 0 or greater.")
            match file_reader:
                case "simple":
                    self.logger.info(f"Using simple reader for file `{file_name}`")
                    file_content = self.get_simple_content_from_file(file, file_name)
                case "ocr":
                    self.logger.info(f"Using ocr reader for file `{file_name}`")
                    file_content = self.get_ocr_content_from_file(file, file_name)
                case "markdown":
                    self.logger.info(f"Using markdown reader for file `{file_name}`")
                    file_content = self.get_markdown_content_from_file(file, file_name)
                case _:
                    self.logger.error(f"Unsupported file reader option: {file_reader}")
                    raise ValueError(f"Unsupported file reader: {file_reader}")
            splitter = self.get_splitter(model_name, chunk_size, chunk_overlap)
            chunks = splitter.split_text(file_content)
            self.logger.info(f"Successfully split file `{file_name}` into {len(chunks)} using model `{model_name}`")
            return chunks
        except Exception as e:
            self.logger.exception(f"Error during chunking process for file `{file_name}`: {e}")
            raise
