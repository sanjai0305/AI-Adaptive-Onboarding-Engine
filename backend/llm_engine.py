"""
backend/llm_engine.py
---------------------
LLMEngine — static façade used by app.py.

All calls delegate to ModelInference (inference.py) which handles
all backend routing. Falls back to rule-based logic when no LLM
is configured.

Backends now supported
----------------------
  openai       — GPT-3.5-turbo / GPT-4 / GPT-4o
  claude       — Anthropic Claude 3 Haiku / Sonnet / Opus
  gemini       — Google Gemini 1.5 Flash / Pro  (NEW)
  ollama       — Local Ollama server  (llama3, mistral, phi3 …)
  huggingface  — Any HuggingFace text-generation model
  local        — Fine-tuned LoRA / full checkpoint
  fallback     — Rule-based (no API key needed)

Factory functions
-----------------
  create_openai_engine(api_key, model)
  create_claude_engine(api_key, model)
  create_gemini_engine(api_key, model)   ← NEW
  create_ollama_engine(model, base_url)
  create_huggingface_engine(model_name)
  create_local_engine(checkpoint_path, use_lora)
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Lazy singleton ──────────────────────────────────────────
_model_inference: Any = None


def _get_model_inference() -> Any:
    global _model_inference
    if _model_inference is None:
        try:
            from inference import ModelInference   # type: ignore
            _model_inference = ModelInference(model_backend="auto")
            logger.info(f"[LLMEngine] Auto-detected backend: {_model_inference.backend}")
        except Exception as exc:
            logger.warning(f"[LLMEngine] ModelInference unavailable: {exc}")
    return _model_inference


# ══════════════════════════════════════════════════════════
class LLMEngine:
    """
    Static/class-method façade so app.py can call:
        LLMEngine.detect_experience_level(...)
        LLMEngine.generate_learning_roadmap(...)
        LLMEngine.generate_resume_suggestions(...)
        LLMEngine.generate_skill_roadmap(...)
    without instantiation.
    """

    @classmethod
    def _backend(cls) -> Any:
        return _get_model_inference()

    # ----------------------------------------------------------
    @classmethod
    def detect_experience_level(cls, resume_text: str, skill: str) -> str:
        """Return Beginner | Intermediate | Advanced | Expert."""
        try:
            m = cls._backend()
            if m:
                return m.detect_experience_level(resume_text, skill)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.error(f"[LLMEngine] detect_experience_level: {exc}")
        return cls._heuristic_level(resume_text, skill)

    # ----------------------------------------------------------
    @classmethod
    def generate_learning_roadmap(
        cls,
        missing_skills: list[dict[str, Any]],
        weak_skills:    list[dict[str, Any]],
        jd_text:        str = "",
    ) -> dict[str, dict[str, Any]]:
        """Return curated resources per missing/weak skill."""
        try:
            m = cls._backend()
            if m:
                return m.generate_learning_roadmap(missing_skills, weak_skills, jd_text)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.error(f"[LLMEngine] generate_learning_roadmap: {exc}")
        return cls._fallback_resources(missing_skills + weak_skills)

    # ----------------------------------------------------------
    @classmethod
    def generate_resume_suggestions(
        cls, resume_text: str, jd_text: str,
    ) -> list[str]:
        """Return a list of actionable resume improvement suggestions."""
        try:
            m = cls._backend()
            if m:
                return m.generate_resume_suggestions(resume_text, jd_text)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.error(f"[LLMEngine] generate_resume_suggestions: {exc}")
        return [
            "Quantify achievements with metrics (%, $, time saved).",
            "Mirror keywords from the job description.",
            "Add a concise professional summary.",
            "List projects with GitHub links or live demos.",
            "Highlight leadership and collaboration examples.",
            "Tailor skills section to the exact tools in the JD.",
        ]

    # ----------------------------------------------------------
    @classmethod
    def generate_skill_roadmap(
        cls,
        skill:         str,
        current_level: str = "beginner",
        target_level:  str = "advanced",
        weeks:         int = 8,
    ) -> dict[str, Any]:
        """Return a week-by-week roadmap for a single skill."""
        try:
            m = cls._backend()
            if m:
                return m.generate_roadmap(skill, current_level, target_level, weeks)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.error(f"[LLMEngine] generate_skill_roadmap: {exc}")
        return cls._fallback_roadmap(skill, current_level, target_level, weeks)

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------
    @staticmethod
    def _heuristic_level(text: str, skill: str) -> str:
        tl     = text.lower()
        count  = tl.count(skill.lower())
        senior = sum(tl.count(w) for w in ["senior", "lead", "principal", "expert", "5+ years"])
        mid    = sum(tl.count(w) for w in ["2 years", "3 years", "intermediate", "mid-level"])
        if count == 0:                return "Beginner"
        if senior >= 2 or count >= 5: return "Advanced"
        if mid    >= 1 or count >= 2: return "Intermediate"
        return "Beginner"

    @staticmethod
    def _fallback_resources(skills: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            s.get("skill", str(s)): {
                "subtopics":       ["Fundamentals", "Intermediate", "Advanced"],
                "resources":       [f"https://www.google.com/search?q=learn+{s.get('skill', 'skill')}"],
                "projects":        ["Build a small practice project"],
                "estimated_hours": 30,
            }
            for s in skills if isinstance(s, dict)
        }

    @staticmethod
    def _fallback_roadmap(
        skill: str, current: str, target: str, weeks: int,
    ) -> dict[str, Any]:
        phases = ["Foundations", "Core Concepts", "Intermediate", "Advanced", "Projects"]
        plan = [
            {
                "week":      w,
                "topics":    [f"{skill} — {phases[min((w-1)*len(phases)//weeks, len(phases)-1)]}"],
                "resources": [f"https://www.google.com/search?q=learn+{skill.replace(' ', '+')}"],
                "project":   f"Practice project — week {w}",
                "hours":     10,
            }
            for w in range(1, weeks + 1)
        ]
        return {
            "skill": skill, "current_level": current, "target_level": target,
            "duration_weeks": weeks, "weekly_plan": plan,
            "estimated_hours": weeks * 10, "priority": "high",
        }


# ══════════════════════════════════════════════════════════
# Factory functions — switch backend at runtime
# Return type is `type[LLMEngine]` (the class, not an instance)
# so callers can still use LLMEngine.method() after switching.
# ══════════════════════════════════════════════════════════

def create_openai_engine(
    api_key: str,
    model:   str = "gpt-3.5-turbo",
) -> type[LLMEngine]:
    """Switch global backend to OpenAI GPT."""
    from inference import ModelInference           # type: ignore
    global _model_inference
    _model_inference = ModelInference(
        model_backend="openai", api_key=api_key, openai_model=model,
    )
    logger.info(f"[LLMEngine] → OpenAI (model={model})")
    return LLMEngine


def create_claude_engine(
    api_key: str,
    model:   str = "claude-3-haiku-20240307",
) -> type[LLMEngine]:
    """Switch global backend to Anthropic Claude."""
    from inference import ModelInference           # type: ignore
    global _model_inference
    _model_inference = ModelInference(
        model_backend="claude", api_key=api_key, claude_model=model,
    )
    logger.info(f"[LLMEngine] → Claude (model={model})")
    return LLMEngine


def create_gemini_engine(
    api_key: str,
    model:   str = "gemini-1.5-flash",
) -> type[LLMEngine]:
    """
    Switch global backend to Google Gemini.

    Parameters
    ----------
    api_key : str
        Google AI Studio API key (starts with 'AIza…').
        Get one at https://aistudio.google.com/app/apikey
    model : str
        Gemini model name.
        Options: 'gemini-1.5-flash' (fast) | 'gemini-1.5-pro' (smart)
                 'gemini-pro' (legacy)

    Example
    -------
        from backend.llm_engine import create_gemini_engine
        create_gemini_engine(api_key="AIza...", model="gemini-1.5-pro")
    """
    from inference import ModelInference           # type: ignore
    global _model_inference
    _model_inference = ModelInference(
        model_backend="gemini",
        gemini_api_key=api_key,
        gemini_model=model,
    )
    logger.info(f"[LLMEngine] → Gemini (model={model})")
    return LLMEngine


def create_ollama_engine(
    model:    str = "llama3",
    base_url: str = "http://localhost:11434",
) -> type[LLMEngine]:
    """
    Switch global backend to a local Ollama server.

    Parameters
    ----------
    model : str
        Model tag as shown in `ollama list`.
        Popular options: llama3, mistral, phi3, gemma2, qwen2,
                         codellama, deepseek-coder, llava (multimodal)
    base_url : str
        Ollama server URL (default http://localhost:11434).

    Prerequisites
    -------------
        brew install ollama   # macOS
        ollama serve          # start server
        ollama pull llama3    # download model
    """
    from inference import ModelInference           # type: ignore
    global _model_inference
    _model_inference = ModelInference(
        model_backend="ollama",
        ollama_model=model,
        ollama_base_url=base_url,
    )
    logger.info(f"[LLMEngine] → Ollama (model={model}, url={base_url})")
    return LLMEngine


def create_huggingface_engine(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> type[LLMEngine]:
    """Switch global backend to a HuggingFace model (downloaded locally)."""
    from inference import ModelInference           # type: ignore
    global _model_inference
    _model_inference = ModelInference(
        model_backend="huggingface", hf_model_name=model_name,
    )
    logger.info(f"[LLMEngine] → HuggingFace (model={model_name})")
    return LLMEngine


def create_local_engine(
    checkpoint_path: str,
    use_lora:        bool = False,
) -> type[LLMEngine]:
    """Switch global backend to a local fine-tuned checkpoint."""
    from inference import ModelInference           # type: ignore
    global _model_inference
    _model_inference = ModelInference(
        checkpoint_path=checkpoint_path,
        use_lora=use_lora,
        model_backend="local",
    )
    logger.info(f"[LLMEngine] → Local checkpoint (path={checkpoint_path}, lora={use_lora})")
    return LLMEngine