"""Extract raw text from an uploaded resume (PDF / DOCX / TXT)."""
"""
============================================================
SmartHire AI/ML Project
Resume Parser Module
============================================================

This module extracts text from uploaded resumes.

Supported Formats:
- PDF (.pdf)
- Word Document (.docx)
"""

import pdfplumber
from docx import Document


# ============================================================
# Extract Text from PDF
# ============================================================

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF resume.
    """

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# ============================================================
# Extract Text from DOCX
# ============================================================

def extract_text_from_docx(docx_path):
    """
    Extract text from a DOCX resume.
    """

    document = Document(docx_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )

    return text


# ============================================================
# Universal Resume Loader
# ============================================================

def extract_resume_text(file_path):
    """
    Automatically detect file type and extract text.
    """

    file_path = str(file_path)

    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)

    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)

    else:
        raise ValueError(
            "Unsupported file format. Please upload a PDF or DOCX file."
        )