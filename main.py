import os
import pymupdf
import pandas as pd
from typing import Literal
from pathlib import Path
from transformers import AutoTokenizer
from fastapi import FastAPI, UploadFile, File, Form
from langchain_text_splitters import TokenTextSplitter

from fastapi import FastAPI

title = "LonelyLittleMeowSounds"

app = FastAPI(title=f"{title}", version="0.0.1")

@app.get("/")
def read_root():
    return {
        "message": f"{title} API",
        "usage": "For full API documentation and testing, visit /docs (Swagger UI) or /redoc (ReDoc)."
    }

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

@app.post("/chunking")
def chunking(
    file: UploadFile = File(...), 
    chunk_size: int = Form(500), 
    chunk_overlap: int = Form(20),
    embedding_model: Literal["text-embedding-ada-002", "BAAI/bge-m3"] = Form("text-embedding-ada-002")
):
    try:
        tmp_dir = 'tmp_file/'
        os.makedirs(tmp_dir, exist_ok=True)
        file_path = os.path.join(tmp_dir, file.filename)
        with open(file_path, 'wb') as f:
            f.write(file.file.read())
        file_ext = Path(file_path).suffix.lower()
        content = ""
        match file_ext:
            case ".csv":
                df = pd.read_csv(file_path)
                content = df.to_string(index=False)
            case ".xls" | ".xlsx":
                df = pd.read_excel(file_path)
                content = df.to_string(index=False)
            case _:
                doc = pymupdf.open(file_path)
                content = "".join(page.get_text() for page in doc)
        os.remove(file_path)
        match embedding_model:
            case "text-embedding-ada-002":
                splitter = TokenTextSplitter.from_tiktoken_encoder(
                    model_name=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            case "BAAI/bge-m3":
                splitter = TokenTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(content)

        return {"data": chunks}
    except Exception as e:
        return {"error": str(e)}