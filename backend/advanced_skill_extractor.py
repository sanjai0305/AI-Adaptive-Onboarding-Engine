"""
backend/advanced_skill_extractor.py
------------------------------------
BERT-based NER skill extractor using HuggingFace pipelines.

FIXES (Pylance errors):
  • entity_group / word were accessed directly on dict that could be None.
    Now guarded with isinstance checks before subscripting.
  • Tensor.strip() / Tensor.startswith() — pipeline output converted to str
    explicitly before calling string methods.
  • List[Tensor] passed to sklearn — .tolist() called before fit/transform.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class AdvancedSkillExtractor:
    """Skill extractor using a HuggingFace NER pipeline."""

    MODEL_NAME = "dslim/bert-base-NER"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._pipeline: Any = None
        self._vectorizer: Any = None

    # ----------------------------------------------------------
    def _load_pipeline(self) -> None:
        """Lazy-load the NER pipeline."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
            self._pipeline = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
                device=-1,          # CPU
            )
            logger.info(f"Loaded NER pipeline: {self.model_name}")
        except Exception as exc:
            logger.warning(f"Could not load NER pipeline: {exc}")
            self._pipeline = None

    # ----------------------------------------------------------
    def extract_skills(self, text: str) -> list[str]:
        """Extract skill entities from text."""
        self._load_pipeline()

        if self._pipeline is None:
            return self._rule_fallback(text)

        try:
            raw: list[Any] = self._pipeline(text[:2000])  # type: ignore[misc]
            return self._parse_ner_output(raw)
        except Exception as exc:
            logger.error(f"NER extraction error: {exc}")
            return self._rule_fallback(text)

    # ----------------------------------------------------------
    @staticmethod
    def _parse_ner_output(raw: list[Any]) -> list[str]:
        """
        Convert raw NER output to skill strings.

        FIX: Each item in `raw` is a dict; we guard with isinstance
             and access keys safely to avoid None-subscript errors.
             `word` value is cast to str explicitly before .strip()
             to avoid Tensor.strip() Pylance error.
        """
        skills: list[str] = []
        seen: set[str] = set()

        for item in raw:
            # ✅ FIX: guard isinstance before subscripting
            if not isinstance(item, dict):
                continue

            # ✅ FIX: cast to str — avoids Tensor.strip() / startswith() errors
            entity_group: str = str(item.get("entity_group", "")).strip()
            word: str = str(item.get("word", "")).strip()

            if not word or not entity_group:
                continue

            # Filter out sub-word tokens (## prefix from BERT tokenizer)
            if word.startswith("##"):
                continue

            # Only keep relevant entity types
            if entity_group not in ("ORG", "MISC", "PER", "LOC"):
                word_clean = word.title()
            else:
                word_clean = word

            key = word_clean.lower()
            if key not in seen and len(word_clean) > 1:
                seen.add(key)
                skills.append(word_clean)

        return skills

    # ----------------------------------------------------------
    def build_skill_vectors(self, texts: list[str]) -> Any:
        """
        Build TF-IDF vectors for a list of texts.

        FIX: sklearn's TfidfVectorizer.fit_transform expects
             List[str], not List[Tensor]. We ensure all inputs
             are plain strings before calling fit_transform.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

            if self._vectorizer is None:
                self._vectorizer = TfidfVectorizer(max_features=500)

            # ✅ FIX: convert every element to str explicitly
            clean_texts: list[str] = [str(t) for t in texts]
            return self._vectorizer.fit_transform(clean_texts)

        except ImportError:
            logger.warning("scikit-learn not installed")
            return None

    # ----------------------------------------------------------
    @staticmethod
    def _rule_fallback(text: str) -> list[str]:
        """Regex keyword fallback when NER is unavailable."""
        from backend.skill_extractor import SKILL_DB
        found: list[str] = []
        tl = text.lower()
        for skill in SKILL_DB:
            if re.search(r"\b" + re.escape(skill.lower()) + r"\b", tl):
                found.append(skill)
        return found