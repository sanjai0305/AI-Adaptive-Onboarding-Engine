"""
backend/parser.py
-----------------
Parse text from PDF and DOCX files.

FIX: Pylance flagged `fitz.open` as unknown attribute.
     Resolution: use `fitz.Document(path)` which is the
     public constructor, or fall back to `io.BytesIO` path.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class FileParser:
    """Parse PDF and DOCX files into plain text."""

    # ----------------------------------------------------------
    @staticmethod
    def parse_file(file_path: str) -> str:
        """
        Auto-detect format from extension and return raw text.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to a .pdf or .docx file.

        Returns
        -------
        str
            Extracted plain text.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return FileParser._parse_pdf(file_path)
        elif ext == ".docx":
            return FileParser._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ----------------------------------------------------------
    @staticmethod
    def _parse_pdf(path: str) -> str:
        """
        Extract text from a PDF using PyMuPDF (fitz).

        FIX: `fitz.Document(path)` is the correct public constructor.
             `fitz.open(path)` is an alias — Pylance does not
             recognise it from stubs; `fitz.Document` is always safe.
        """
        try:
            import fitz  # PyMuPDF

            # ✅ FIX: use fitz.Document instead of fitz.open
            doc: fitz.Document = fitz.Document(path)
            pages: list[str] = []
            for page in doc:
                pages.append(page.get_text())  # type: ignore[attr-defined]
            doc.close()
            return "\n".join(pages)

        except ImportError:
            logger.warning("PyMuPDF not installed. Falling back to pdfminer.")
            return FileParser._parse_pdf_fallback(path)

        except Exception as exc:
            logger.error(f"PDF parse error for {path}: {exc}")
            raise

    @staticmethod
    def _parse_pdf_fallback(path: str) -> str:
        """Fallback PDF parser using pdfminer.six."""
        try:
            from pdfminer.high_level import extract_text  # type: ignore
            return extract_text(path)
        except ImportError:
            raise ImportError(
                "Neither PyMuPDF nor pdfminer.six is installed.\n"
                "Run: pip install pymupdf  OR  pip install pdfminer.six"
            )

    # ----------------------------------------------------------
    @staticmethod
    def _parse_docx(path: str) -> str:
        """Extract text from a DOCX file using python-docx."""
        try:
            import docx  # type: ignore

            doc = docx.Document(path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            paragraphs.append(cell.text.strip())
            return "\n".join(paragraphs)

        except ImportError:
            raise ImportError(
                "python-docx not installed.\nRun: pip install python-docx"
            )
        except Exception as exc:
            logger.error(f"DOCX parse error for {path}: {exc}")
            raise

    # ----------------------------------------------------------
    @staticmethod
    def parse_bytes(data: bytes, filename: str) -> str:
        """Parse from raw bytes (useful for Streamlit UploadedFile)."""
        ext = os.path.splitext(filename)[1].lower()
        tmp = f"/tmp/_parser_{os.getpid()}{ext}"
        with open(tmp, "wb") as f:
            f.write(data)
        try:
            return FileParser.parse_file(tmp)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass