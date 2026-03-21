"""
backend/matcher.py
------------------
Semantic skill matching using sentence embeddings.

FIXES (Pylance errors):
  • cosine_similarity expected Tensor, not List[Tensor].
    Resolution: stack list into a single tensor with torch.stack()
    before calling util.cos_sim, or convert to numpy array first.
  • Overload __getitem__ mismatch: use .item() to extract scalar float.
"""
from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────
MATCHED_THRESHOLD = 0.75
WEAK_THRESHOLD    = 0.50


class SkillMatch(TypedDict):
    skill:      str
    match:      str
    similarity: float


class SkillMatcher:
    """Match resume skills against JD skills semantically."""

    _model: object = None   # lazy-loaded SentenceTransformer

    # ----------------------------------------------------------
    @classmethod
    def _get_model(cls) -> object:
        if cls._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                cls._model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded SentenceTransformer: all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed — using TF-IDF fallback")
        return cls._model

    # ----------------------------------------------------------
    @classmethod
    def match_skills(
        cls,
        resume_skills: list[str],
        jd_skills:     list[str],
    ) -> tuple[list[SkillMatch], list[SkillMatch], list[SkillMatch]]:
        """
        Classify JD skills as Matched / Weak / Missing against resume.

        Returns
        -------
        (matched, missing, weak) — each a list of SkillMatch dicts.
        """
        if not jd_skills:
            return [], [], []

        model = cls._get_model()

        if model is not None:
            return cls._semantic_match(model, resume_skills, jd_skills)
        else:
            return cls._tfidf_match(resume_skills, jd_skills)

    # ----------------------------------------------------------
    @classmethod
    def _semantic_match(
        cls,
        model: object,
        resume_skills: list[str],
        jd_skills:     list[str],
    ) -> tuple[list[SkillMatch], list[SkillMatch], list[SkillMatch]]:
        """Sentence-transformer cosine similarity matching."""
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
            import torch  # type: ignore

            st_model: SentenceTransformer = model  # type: ignore[assignment]

            # Encode — returns numpy array when convert_to_tensor=False
            resume_embs: np.ndarray = st_model.encode(  # type: ignore[assignment]
                resume_skills, convert_to_tensor=False, show_progress_bar=False
            )
            jd_embs: np.ndarray = st_model.encode(  # type: ignore[assignment]
                jd_skills, convert_to_tensor=False, show_progress_bar=False
            )

            # ✅ FIX: convert numpy → tensor explicitly so cos_sim gets Tensor, not list
            resume_tensor = torch.tensor(resume_embs, dtype=torch.float32)
            jd_tensor     = torch.tensor(jd_embs,     dtype=torch.float32)

            # cos_sim returns a 2-D Tensor [n_jd x n_resume]
            sim_matrix = util.cos_sim(jd_tensor, resume_tensor)

            matched: list[SkillMatch] = []
            weak:    list[SkillMatch] = []
            missing: list[SkillMatch] = []

            for j, jd_skill in enumerate(jd_skills):
                row = sim_matrix[j]                 # 1-D tensor length n_resume
                if len(resume_skills) == 0:
                    best_score = 0.0
                    best_idx   = 0
                else:
                    # ✅ FIX: use .item() to get Python float, not Tensor
                    best_idx   = int(row.argmax().item())
                    best_score = float(row[best_idx].item())

                best_match = resume_skills[best_idx] if resume_skills else ""

                entry: SkillMatch = {
                    "skill":      jd_skill,
                    "match":      best_match,
                    "similarity": round(best_score, 4),
                }

                if best_score >= MATCHED_THRESHOLD:
                    matched.append(entry)
                elif best_score >= WEAK_THRESHOLD:
                    weak.append(entry)
                else:
                    missing.append(entry)

            return matched, missing, weak

        except Exception as exc:
            logger.error(f"Semantic match failed: {exc}")
            return cls._tfidf_match(resume_skills, jd_skills)

    # ----------------------------------------------------------
    @staticmethod
    def _tfidf_match(
        resume_skills: list[str],
        jd_skills:     list[str],
    ) -> tuple[list[SkillMatch], list[SkillMatch], list[SkillMatch]]:
        """
        TF-IDF cosine-similarity fallback when sentence-transformers
        is unavailable.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.metrics.pairwise import cosine_similarity        # type: ignore

            all_skills = resume_skills + jd_skills
            vec = TfidfVectorizer().fit(all_skills)

            # ✅ FIX: pass plain list[str] — no Tensor involved
            resume_vecs = vec.transform(resume_skills)
            jd_vecs     = vec.transform(jd_skills)

            sim_matrix: np.ndarray = cosine_similarity(jd_vecs, resume_vecs)

        except ImportError:
            # Pure keyword fallback
            resume_set = {s.lower() for s in resume_skills}
            sim_matrix = np.zeros((len(jd_skills), max(len(resume_skills), 1)))
            for j, jd_skill in enumerate(jd_skills):
                if jd_skill.lower() in resume_set:
                    sim_matrix[j, 0] = 1.0

        matched: list[SkillMatch] = []
        weak:    list[SkillMatch] = []
        missing: list[SkillMatch] = []

        for j, jd_skill in enumerate(jd_skills):
            row = sim_matrix[j]
            if len(resume_skills) == 0:
                best_score, best_idx = 0.0, 0
            else:
                best_idx   = int(np.argmax(row))
                best_score = float(row[best_idx])

            best_match = resume_skills[best_idx] if resume_skills else ""
            entry: SkillMatch = {
                "skill":      jd_skill,
                "match":      best_match,
                "similarity": round(best_score, 4),
            }

            if best_score >= MATCHED_THRESHOLD:
                matched.append(entry)
            elif best_score >= WEAK_THRESHOLD:
                weak.append(entry)
            else:
                missing.append(entry)

        return matched, missing, weak