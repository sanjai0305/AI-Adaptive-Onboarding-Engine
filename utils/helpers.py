"""utils/helpers.py — Text cleaning and general utilities."""
import re
import unicodedata


class TextCleaner:
    """Static helpers for normalising raw document text."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove noise from extracted PDF/DOCX text."""
        if not text:
            return ""
        # Normalise unicode
        text = unicodedata.normalize("NFKD", text)
        # Collapse excessive whitespace / newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Strip control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()

    @staticmethod
    def extract_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    def normalise_skill(skill: str) -> str:
        """Title-case and strip a skill name."""
        return skill.strip().title()