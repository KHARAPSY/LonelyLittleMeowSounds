import io
from fastapi.testclient import TestClient
from your_module import app  # replace 'your_module' with the filename without '.py'

client = TestClient(app)

# Helper to simulate file uploads
def simulate_upload(file_content: bytes, filename: str, content_type: str = "application/octet-stream"):
    return {
        "file": (filename, io.BytesIO(file_content), content_type),
        "chunk_size": (None, "100"),
        "chunk_overlap": (None, "10"),
        "embedding_model": (None, "BAAI/bge-m3")
    }

def test_chunking_csv():
    csv_data = b"name,age\nAlice,30\nBob,25"
    response = client.post("/chunking", files=simulate_upload(csv_data, "test.csv"))
    assert response.status_code == 200
    assert "data" in response.json()
    assert isinstance(response.json()["data"], list)

def test_chunking_excel():
    # Create an in-memory Excel file
    import pandas as pd
    from io import BytesIO

    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)

    response = client.post("/chunking", files=simulate_upload(excel_buffer.read(), "test.xlsx"))
    assert response.status_code == 200
    assert "data" in response.json()
    assert isinstance(response.json()["data"], list)

def test_chunking_pdf():
    # Create a simple PDF in-memory
    import fitz  # PyMuPDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello, this is a test PDF.")
    pdf_bytes = doc.write()
    doc.close()

    response = client.post("/chunking", files=simulate_upload(pdf_bytes, "test.pdf"))
    assert response.status_code == 200
    assert "data" in response.json()
    assert isinstance(response.json()["data"], list)

def test_invalid_file():
    response = client.post("/chunking", files=simulate_upload(b"Not really a file", "fake.txt"))
    assert response.status_code == 200
    assert "data" in response.json() or "error" in response.json()
