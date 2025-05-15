# LonelyLittleMeowSounds
LonelyLittleMeowSounds is your friendly companion for harnessing large language models to make daily tasks smarter and more efficient. This Python package provides tools to extract text from various file formats (e.g., PDF, CSV, Excel) and split it into chunks suitable for language model processing, with support for multiple tokenizers and file readers.

## Features

* Text Extraction: Extract plain text from files like PDF, CSV, Excel using pymupdf and pandas.
* OCR Support: Extract text from image-based PDFs using pytesseract.
* Markdown Conversion: Convert documents (PDF, DOCX, etc.) to Markdown using docling.
* Text Chunking: Split text into chunks for language models with customizable chunk size and overlap.
* Tokenizer Support: Use tokenizers from Hugging Face models or OpenAI's tiktoken for text splitting.
* Multilingual Models: Preload tokenizers for popular models like [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct), [gte-Qwen2-1.5B-instruct](Alibaba-NLP/gte-Qwen2-1.5B-instruct), [bilingual-embedding-large](https://huggingface.co/Lajavaness/bilingual-embedding-large), [bge-m3](https://huggingface.co/BAAI/bge-m3), [snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) and more.

## Installation
Follow these steps to install LonelyLittleMeowSounds from source by cloning the GitHub repository.

### Prerequisites

* Python 3.10 or higher
* [git](https://git-scm.com/downloads)
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract)

### Steps

1. Clone the Repository
```bash
git clone https://github.com/KHARAPSY/LonelyLittleMeowSounds.git
cd LonelyLittleMeowSounds
```

2. Create a Virtual Environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install Dependencies
Install the required Python packages listed in [requirements.txt](requirements.txt):
```bash
pip install -r requirements.txt
```

4. Install the Package
Install [LonelyLittleMeowSounds](https://github.com/KHARAPSY/LonelyLittleMeowSounds) in editable mode:
```bash
pip install -e .
```

5. Verify Installation
Test the installation by running:
```bash
python
```
In the Python shell:
```python
from lolimeowss import LLMeowsSChunks
llms = LLMeowsSChunks()
llms.hi_meows()
```
Expected output:
```plain
Meowss, meowss is your pal-ai ready to help!
```

## Usage
The `LLMeowsSChunks` class provides methods to extract and process text from files. Below are some examples.

#### Example 1: Extract Text from a PDF
Extract plain text from a PDF file:
```python
from lolimeowss import LLMeowsSChunks

llms = LLMeowsSChunks()
with open("example.pdf", "rb") as file:
    content = llms.get_simple_content_from_file(file, "example.pdf")
    print(content)
```

#### Example 2: Extract Text with OCR
Extract text from an image-based PDF using OCR:
```python
from lolimeowss import LLMeowsSChunks

llms = LLMeowsSChunks()
with open("image-based.pdf", "rb") as file:
    content = llms.get_ocr_content_from_file(file, "image-based.pdf")
    print(content)
```

#### Example 3: Chunk Text for Language Models
Split a PDF's content into chunks for a specific model:
```python
from lolimeowss import LLMeowsSChunks

llms = LLMeowsSChunks()
with open("example.pdf", "rb") as file:
    chunks = llms.get_chunking(
        file=file,
        file_name="example.pdf",
        model_name="intfloat/multilingual-e5-large-instruct",
        chunk_size=500,
        chunk_overlap=25,
        file_reader="simple"
    )
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:100]}...")  # Print first 100 chars of each chunk
```

### Supported File Readers
* `simple`: Extracts plain text (PDF, CSV, Excel).
* `ocr`: Extracts text from image-based PDFs using OCR.
* `markdown`: Converts documents to Markdown (PDF, DOCX, Excel, HTML, PPTX).

## Dependencies
The following Python packages are required (listed in [requirements.txt](requirements.txt)):
* `pymupdf`: For PDF text extraction
* `pandas`: For CSV and Excel processing
* `transformers[torch]`: For Hugging Face tokenizers
* `langchain-text-splitters`: For text chunking
* `python-multipart`: For file handling
* `tiktoken`: For OpenAI model tokenization
* `sphinx`, `furo`: For documentation
* `pytest`: For testing
* `httpx`: For HTTP requests
* `docling`: For Markdown conversion

Some dependencies (e.g., [pytesseract](https://github.com/madmaze/pytesseract)) require system-level installations. For example, on Ubuntu:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-tha
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature).
3. Make your changes and commit (git commit -m "Add your feature").
4. Push to your branch (git push origin feature/your-feature).
5. Open a Pull Request on GitHub.

Please include tests and update documentation as needed.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
* GitHub: [KHARAPSY](https://github.com/KHARAPSY)
* X: [@KHARAPSY](https://x.com/kharapsy)
* Documentation: [documentation](https://lonelylittlemeowsounds.suwalutions.com/documentation)
* Issues: [GitHub Issues](https://github.com/KHARAPSY/LonelyLittleMeowSounds/issues)

Meowss, meowss! Let's make tasks smarter together!
