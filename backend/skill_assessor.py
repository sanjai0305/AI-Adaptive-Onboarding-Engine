"""
backend/skill_assessor.py
--------------------------
AI-generated skill assessment quizzes.

FIX: `choices` attribute was accessed on a Generator object returned
     by the OpenAI client; now properly accessed after awaiting/calling
     the response and with safe attribute access.
"""
from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Question:
    """A single quiz question."""

    def __init__(
        self,
        question: str,
        options:  list[str],
        correct:  str,
        explanation: str = "",
    ) -> None:
        self.question    = question
        self.options     = options
        self.correct     = correct
        self.explanation = explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "question":    self.question,
            "options":     self.options,
            "correct":     self.correct,
            "explanation": self.explanation,
        }


class SkillAssessor:
    """Generate and score skill assessment quizzes."""

    # ----------------------------------------------------------
    @staticmethod
    def generate_quiz(
        skill:      str,
        difficulty: str = "intermediate",
        n_questions: int = 5,
    ) -> list[Question]:
        """
        Generate quiz questions for a skill.

        Tries LLM first; falls back to template-based questions.
        """
        try:
            questions = SkillAssessor._llm_questions(skill, difficulty, n_questions)
            if questions:
                return questions
        except Exception as exc:
            logger.warning(f"LLM quiz generation failed: {exc}")

        return SkillAssessor._template_questions(skill, n_questions)

    # ----------------------------------------------------------
    @staticmethod
    def _llm_questions(
        skill:      str,
        difficulty: str,
        n:          int,
    ) -> list[Question]:
        """Generate questions via OpenAI API."""
        try:
            from openai import OpenAI  # type: ignore

            import os
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                return []

            client = OpenAI(api_key=api_key)

            prompt = (
                f"Generate {n} multiple-choice quiz questions about '{skill}' "
                f"at {difficulty} level. "
                "Return JSON array: "
                '[{"question":"...","options":["A","B","C","D"],"correct":"A","explanation":"..."}]'
            )

            # ✅ FIX: response is a ChatCompletion object, not a Generator
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.4,
            )

            # ✅ FIX: safe attribute access — choices[0].message.content
            if not response.choices:
                return []

            raw_text: str = response.choices[0].message.content or ""
            return SkillAssessor._parse_questions(raw_text)

        except Exception as exc:
            logger.error(f"OpenAI quiz error: {exc}")
            return []

    # ----------------------------------------------------------
    @staticmethod
    def _parse_questions(raw: str) -> list[Question]:
        """Parse JSON quiz response into Question objects."""
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                return []
            questions: list[Question] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = Question(
                    question    = str(item.get("question", "")),
                    options     = [str(o) for o in item.get("options", [])],
                    correct     = str(item.get("correct", "")),
                    explanation = str(item.get("explanation", "")),
                )
                if q.question and q.options:
                    questions.append(q)
            return questions
        except (json.JSONDecodeError, Exception):
            return []

    # ----------------------------------------------------------
    @staticmethod
    def _template_questions(skill: str, n: int) -> list[Question]:
        """Fallback template-based questions (no LLM required)."""
        templates = [
            Question(
                question=f"Which of the following best describes {skill}?",
                options=[
                    f"A framework for building {skill} applications",
                    f"A programming language called {skill}",
                    f"A tool or technology used in {skill}",
                    f"A database related to {skill}",
                ],
                correct=f"A tool or technology used in {skill}",
                explanation=f"{skill} is a widely-used tool or technology.",
            ),
            Question(
                question=f"What is a common use case for {skill}?",
                options=[
                    "Data processing and transformation",
                    "Network hardware configuration",
                    "Physical server maintenance",
                    "Document formatting",
                ],
                correct="Data processing and transformation",
                explanation=f"{skill} is commonly used for data processing.",
            ),
            Question(
                question=f"Which skill is closely related to {skill}?",
                options=["Python", "COBOL", "Assembly", "Pascal"],
                correct="Python",
                explanation="Python pairs well with most modern tech stacks.",
            ),
        ]
        return templates[:n]

    # ----------------------------------------------------------
    @staticmethod
    def score_quiz(
        questions: list[Question],
        answers:   dict[int, str],
    ) -> dict[str, Any]:
        """
        Score a completed quiz.

        Parameters
        ----------
        questions : list of Question objects
        answers   : {question_index: user_answer_string}

        Returns
        -------
        {"score": int, "total": int, "percentage": float, "results": [...]}
        """
        total   = len(questions)
        correct = 0
        results: list[dict[str, Any]] = []

        for idx, q in enumerate(questions):
            user_ans  = answers.get(idx, "")
            is_correct = user_ans.strip().lower() == q.correct.strip().lower()
            if is_correct:
                correct += 1
            results.append({
                "question":    q.question,
                "user_answer": user_ans,
                "correct":     q.correct,
                "is_correct":  is_correct,
                "explanation": q.explanation,
            })

        pct = (correct / total * 100) if total > 0 else 0.0
        return {
            "score":      correct,
            "total":      total,
            "percentage": round(pct, 1),
            "passed":     pct >= 70,
            "results":    results,
        }