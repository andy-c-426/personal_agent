import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def sample_md_file(temp_dir):
    p = temp_dir / "test.md"
    p.write_text("# Test Doc\n\nThis is a test document for knowledge base ingestion.")
    return p


@pytest.fixture
def sample_pdf_file(temp_dir):
    p = temp_dir / "test.pdf"
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, text="Test PDF content for ingestion pipeline")
    pdf.output(str(p))
    return p


@pytest.fixture
def sample_docx_file(temp_dir):
    from docx import Document
    p = temp_dir / "test.docx"
    doc = Document()
    doc.add_paragraph("Test Word document content for ingestion pipeline")
    doc.save(str(p))
    return p


@pytest.fixture
def sample_text_file(temp_dir):
    p = temp_dir / "test.txt"
    p.write_text("Plain text content for the knowledge base.")
    return p
