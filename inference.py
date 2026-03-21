"""
inference.py
============================================================
ModelInference — unified multi-LLM interface.

Supported backends
------------------
  1. openai       — GPT-3.5-turbo / GPT-4 / GPT-4o
  2. claude       — Anthropic Claude 3 Haiku / Sonnet / Opus
  3. gemini       — Google Gemini 1.5 Flash / Pro
  4. ollama       — Local open-source models via Ollama server
                    (LLaMA 3, Mistral, Phi-3, Gemma, Qwen …)
  5. huggingface  — Any HuggingFace text-generation model
  6. local        — Fine-tuned LoRA / full checkpoint
  7. fallback     — Rule-based keyword matching (zero deps)

Quick start
-----------
    from inference import ModelInference

    # Google Gemini
    model = ModelInference(model_backend='gemini',
                           gemini_api_key='AIza...',
                           gemini_model='gemini-1.5-flash')

    # Local Ollama  (make sure `ollama serve` is running)
    model = ModelInference(model_backend='ollama',
                           ollama_model='llama3',
                           ollama_base_url='http://localhost:11434')

    # OpenAI
    model = ModelInference(model_backend='openai', api_key='sk-...')

    # All models share the same interface
    skills  = model.extract_skills(resume_text)
    roadmap = model.generate_roadmap('Python')
    level   = model.detect_experience_level(resume_text, 'Python')
    tips    = model.generate_resume_suggestions(resume_text, jd_text)
    res_map = model.generate_learning_roadmap(missing, weak, jd_text)

Install
-------
    pip install google-generativeai   # Gemini
    pip install openai                # OpenAI
    pip install anthropic             # Claude
    pip install requests              # Ollama (already stdlib-adjacent)
    pip install transformers peft     # HuggingFace / local

Ollama quick-start
------------------
    ollama serve                      # start the server
    ollama pull llama3                # pull a model first!
    ollama list                       # verify it's available
============================================================
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# Optional dependency guards — all heavy libs are lazy
# ══════════════════════════════════════════════════════════

# ── OpenAI ───────────────────────────────────────────────
try:
    from openai import OpenAI as _OpenAI          # type: ignore[import]
    _OPENAI_OK = True
except ImportError:
    _OpenAI    = None                              # type: ignore[assignment,misc]
    _OPENAI_OK = False

# ── Anthropic Claude ─────────────────────────────────────
try:
    import anthropic as _anthropic                 # type: ignore[import]
    _ANTHROPIC_OK = True
except ImportError:
    _anthropic    = None                           # type: ignore[assignment]
    _ANTHROPIC_OK = False

# ── Google Gemini ─────────────────────────────────────────
try:
    import google.generativeai as _genai          # type: ignore[import]
    _GEMINI_OK = True
except ImportError:
    _genai     = None                             # type: ignore[assignment]
    _GEMINI_OK = False

# ── PEFT / LoRA ──────────────────────────────────────────
try:
    from peft import PeftModel, PeftConfig        # type: ignore[import]
    _PEFT_OK = True
except ImportError:
    _PEFT_OK = False


# ══════════════════════════════════════════════════════════
# Skill taxonomy  (rule-based fallback extractor)
# ══════════════════════════════════════════════════════════
SKILL_TAXONOMY: list[str] = [
    # Languages
    "Python","Java","JavaScript","TypeScript","C++","C#","Go","Rust",
    "Kotlin","Swift","R","Scala","PHP","Ruby","Bash","SQL","HTML","CSS",
    # ML / AI
    "Machine Learning","Deep Learning","NLP","Computer Vision",
    "Reinforcement Learning","Statistical Modeling","Data Analysis",
    "Feature Engineering","A/B Testing","LLM","Prompt Engineering",
    "RAG","Fine-tuning","LoRA","Transfer Learning","Generative AI",
    # Frameworks
    "TensorFlow","PyTorch","Keras","Scikit-learn","XGBoost","LightGBM",
    "Hugging Face","FastAPI","Django","Flask","React","Next.js","Node.js",
    "LangChain","LlamaIndex","Streamlit","Gradio","Spring Boot",
    # Data
    "Pandas","NumPy","Matplotlib","Seaborn","Plotly","Tableau","Power BI",
    "PostgreSQL","MySQL","MongoDB","Redis","Elasticsearch","Cassandra",
    "BigQuery","Snowflake","dbt","SQLAlchemy",
    # Cloud / DevOps
    "AWS","Azure","GCP","Docker","Kubernetes","Terraform","CI/CD",
    "Jenkins","GitHub Actions","Linux","Nginx","Ansible","Helm",
    # Data Engineering
    "Apache Spark","Apache Kafka","Airflow","ETL","Hadoop","Flink",
    # Tools
    "Git","GitHub","JIRA","Confluence","Jupyter","VS Code","Postman",
    "OpenAI API","Anthropic API","Gemini API","Vertex AI",
    # Soft skills
    "Communication","Leadership","Team Collaboration","Problem Solving",
    "Project Management","Agile","Scrum","Critical Thinking",
]

SKILL_PREREQUISITES: dict[str, list[str]] = {
    "Machine Learning":   ["Python","Statistics","Linear Algebra"],
    "Deep Learning":      ["Machine Learning","Python"],
    "NLP":                ["Python","Machine Learning"],
    "Computer Vision":    ["Python","Deep Learning"],
    "TensorFlow":         ["Python","Machine Learning"],
    "PyTorch":            ["Python","Machine Learning"],
    "Kubernetes":         ["Docker","Linux"],
    "Apache Spark":       ["Python","SQL"],
    "React":              ["JavaScript","HTML","CSS"],
    "Django":             ["Python","SQL"],
    "FastAPI":            ["Python"],
    "RAG":                ["Python","LangChain"],
    "Fine-tuning":        ["Python","PyTorch"],
    "LangChain":          ["Python"],
    "Airflow":            ["Python","SQL"],
    "Generative AI":      ["Python","Machine Learning","LLM"],
}


# ══════════════════════════════════════════════════════════
# ModelInference
# ══════════════════════════════════════════════════════════
class ModelInference:
    """Unified multi-LLM inference interface."""

    def __init__(
        self,
        # ── local checkpoint ──────────────────────────────
        checkpoint_path: Optional[str] = None,
        use_lora:        bool          = False,
        # ── backend selection ─────────────────────────────
        model_backend:   str           = "auto",
        # ── OpenAI ───────────────────────────────────────
        api_key:         Optional[str] = None,
        openai_model:    str           = "gpt-3.5-turbo",
        # ── Anthropic Claude ──────────────────────────────
        claude_model:    str           = "claude-3-haiku-20240307",
        # ── Google Gemini ────────────────────────────────
        gemini_api_key:  Optional[str] = None,
        gemini_model:    str           = "gemini-1.5-flash",
        # ── Ollama (local open-source) ────────────────────
        ollama_model:    str           = "llama3",
        ollama_base_url: str           = "http://localhost:11434",
        # ── HuggingFace ───────────────────────────────────
        hf_model_name:   str           = "mistralai/Mistral-7B-Instruct-v0.2",
        # ── generation params ─────────────────────────────
        max_tokens:      int           = 1024,
        temperature:     float         = 0.3,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.use_lora        = use_lora
        self.openai_model    = openai_model
        self.claude_model    = claude_model
        self.gemini_model    = gemini_model
        self.ollama_model    = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")   # FIX: strip trailing slash
        self.hf_model_name   = hf_model_name
        self.max_tokens      = max_tokens
        self.temperature     = temperature

        # Resolve API keys (param → env var fallbacks)
        self.api_key        = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        # Internal client handles — all Optional[Any] to satisfy Pylance
        self._hf_pipeline:    Optional[Any] = None
        self._openai_client:  Optional[Any] = None
        self._claude_client:  Optional[Any] = None
        self._gemini_client:  Optional[Any] = None

        # Resolve and initialise backend
        self.backend = self._resolve_backend(model_backend)
        self._load_backend()
        logger.info(f"[ModelInference] active backend='{self.backend}'")

    # ──────────────────────────────────────────────────────
    # Backend resolution
    # ──────────────────────────────────────────────────────
    def _resolve_backend(self, requested: str) -> str:
        """Auto-detect best available backend when 'auto' is requested."""
        if requested != "auto":
            return requested

        # 1. Local checkpoint
        if self.checkpoint_path and os.path.isdir(self.checkpoint_path):
            return "local"

        # 2. OpenAI key
        key = str(self.api_key or "")
        if key.startswith("sk-") and not key.startswith("sk-ant"):
            return "openai"

        # 3. Anthropic key
        if key.startswith("sk-ant"):
            return "claude"

        # 4. Gemini key (starts with "AIza")
        gkey = str(self.gemini_api_key or "")
        if gkey.startswith("AIza") and _GEMINI_OK:
            return "gemini"

        # 5. Ollama server (local, no key)
        #    _validate_ollama_model() will handle auto-selection / fallback
        if self._ollama_server_ok():
            return "ollama"

        return "fallback"

    def _load_backend(self) -> None:
        dispatch = {
            "local":       self._load_local,
            "openai":      self._load_openai,
            "claude":      self._load_claude,
            "gemini":      self._load_gemini,
            "huggingface": self._load_hf,
        }
        loader = dispatch.get(self.backend)
        if loader:
            loader()
        elif self.backend == "ollama":
            # Validate that the requested model is actually pulled
            self._validate_ollama_model()
        else:
            logger.warning("[ModelInference] Rule-based fallback active (no LLM configured)")

    # ──────────────────────────────────────────────────────
    # Ollama helpers (moved up — used during init)
    # ──────────────────────────────────────────────────────
    def _ollama_server_ok(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            import requests                        # type: ignore
            r = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def _validate_ollama_model(self) -> None:
        """
        Check that self.ollama_model is available locally.

        Auto-recovery strategy
        ----------------------
        1. If the requested model IS pulled → use it (happy path).
        2. If the requested model is NOT pulled but OTHER models are available
           → auto-select the first available model and log a warning.
        3. If NO models are pulled at all → log pull instructions and
           fall back to rule-based.
        """
        available = self.list_ollama_models()

        # Normalise names for comparison (strip ":latest" / ":tag")
        def _base(name: str) -> str:
            return name.split(":")[0].lower()

        available_base  = {_base(m): m for m in available}   # base → full name
        requested_base  = _base(self.ollama_model)

        # ── Case 1: requested model is available ──────────────────────────
        if requested_base in available_base:
            # Use the full tag (e.g. "llama3:latest") for the API call
            self.ollama_model = available_base[requested_base]
            logger.info(
                f"[ModelInference] Ollama ready — model='{self.ollama_model}' "
                f"server='{self.ollama_base_url}'"
            )
            return

        # ── Case 2: requested model NOT pulled, but others are available ──
        if available:
            auto_model = available[0]          # pick the first pulled model
            logger.warning(
                f"[ModelInference] Requested model '{self.ollama_model}' is not pulled.\n"
                f"  Auto-selecting '{auto_model}' from available models: {available}\n"
                f"  To use your preferred model:  ollama pull {self.ollama_model}"
            )
            self.ollama_model = auto_model
            logger.info(
                f"[ModelInference] Ollama ready (auto-selected) — "
                f"model='{self.ollama_model}' server='{self.ollama_base_url}'"
            )
            return

        # ── Case 3: no models pulled at all ───────────────────────────────
        logger.error(
            f"[ModelInference] Ollama is running but NO models are pulled.\n"
            f"  Pull any model to get started, for example:\n"
            f"    ollama pull llama3          # Meta LLaMA 3 (8B)\n"
            f"    ollama pull mistral         # Mistral 7B\n"
            f"    ollama pull phi3            # Microsoft Phi-3 (3.8B, fast)\n"
            f"    ollama pull gemma2          # Google Gemma 2 (9B)\n"
            f"    ollama pull qwen2           # Alibaba Qwen2 (7B)\n"
            f"  Browse all models at: https://ollama.ai/library\n"
            f"  Switching to rule-based fallback until a model is pulled."
        )
        self.backend = "fallback"

    # ──────────────────────────────────────────────────────
    # Loader methods
    # ──────────────────────────────────────────────────────
    def _load_local(self) -> None:
        try:
            import torch
            from transformers import (              # type: ignore
                AutoTokenizer, AutoModelForCausalLM, pipeline,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)

            if self.use_lora and _PEFT_OK:
                cfg  = PeftConfig.from_pretrained(self.checkpoint_path)  # type: ignore[possibly-undefined]
                base = AutoModelForCausalLM.from_pretrained(
                    cfg.base_model_name_or_path,
                    torch_dtype=torch.float16, device_map="auto",
                )
                model = PeftModel.from_pretrained(base, self.checkpoint_path)  # type: ignore[possibly-undefined]
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=torch.float16, device_map="auto",
                )

            self._hf_pipeline = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                max_new_tokens=self.max_tokens, temperature=self.temperature,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
            logger.info("[ModelInference] Local checkpoint loaded")
        except Exception as exc:
            logger.error(f"[ModelInference] Local load failed: {exc} — switching to fallback")
            self.backend = "fallback"

    def _load_openai(self) -> None:
        if not _OPENAI_OK:
            logger.warning("[ModelInference] openai not installed — pip install openai")
            self.backend = "fallback"; return
        key = str(self.api_key or os.getenv("OPENAI_API_KEY", ""))
        self._openai_client = _OpenAI(api_key=key)  # type: ignore[misc]
        logger.info(f"[ModelInference] OpenAI ready — model='{self.openai_model}'")

    def _load_claude(self) -> None:
        if not _ANTHROPIC_OK:
            logger.warning("[ModelInference] anthropic not installed — pip install anthropic")
            self.backend = "fallback"; return
        key = str(self.api_key or os.getenv("ANTHROPIC_API_KEY", ""))
        self._claude_client = _anthropic.Anthropic(api_key=key)  # type: ignore[union-attr]
        logger.info(f"[ModelInference] Claude ready — model='{self.claude_model}'")

    def _load_gemini(self) -> None:
        """
        Initialise the Google Gemini client.

        Uses `google-generativeai` SDK.
        Install: pip install google-generativeai
        """
        if not _GEMINI_OK:
            logger.warning(
                "[ModelInference] google-generativeai not installed.\n"
                "  Run: pip install google-generativeai"
            )
            self.backend = "fallback"; return

        key = str(self.gemini_api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""))
        if not key:
            logger.warning("[ModelInference] No Gemini API key provided — switching to fallback")
            self.backend = "fallback"; return

        try:
            _genai.configure(api_key=key)                              # type: ignore[union-attr]
            self._gemini_client = _genai.GenerativeModel(             # type: ignore[union-attr]
                model_name=self.gemini_model,
                generation_config={
                    "temperature":       self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
            logger.info(f"[ModelInference] Gemini ready — model='{self.gemini_model}'")
        except Exception as exc:
            logger.error(f"[ModelInference] Gemini init failed: {exc}")
            self.backend = "fallback"

    def _load_hf(self) -> None:
        try:
            from transformers import pipeline      # type: ignore
            self._hf_pipeline = pipeline(
                "text-generation", model=self.hf_model_name,
                max_new_tokens=self.max_tokens, temperature=self.temperature,
                device_map="auto",
            )
            logger.info(f"[ModelInference] HuggingFace ready — model='{self.hf_model_name}'")
        except Exception as exc:
            logger.error(f"[ModelInference] HuggingFace load failed: {exc}")
            self.backend = "fallback"

    # ──────────────────────────────────────────────────────
    # Core generation dispatcher
    # ──────────────────────────────────────────────────────
    def _generate(self, prompt: str, system: str = "") -> str:
        """Route prompt to the active backend; return response text."""
        try:
            if self.backend == "openai":
                return self._gen_openai(prompt, system)
            if self.backend == "claude":
                return self._gen_claude(prompt, system)
            if self.backend == "gemini":
                return self._gen_gemini(prompt, system)
            if self.backend in ("local", "huggingface"):
                return self._gen_hf(prompt, system)
            if self.backend == "ollama":
                return self._gen_ollama(prompt, system)
        except Exception as exc:
            logger.error(f"[ModelInference] Generation error ({self.backend}): {exc}")
        return ""

    # ── Per-backend generators ────────────────────────────
    def _gen_openai(self, prompt: str, system: str) -> str:
        if self._openai_client is None:
            return ""
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        response = self._openai_client.chat.completions.create(
            model=self.openai_model, messages=msgs,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )
        if not response.choices:
            return ""
        return str(response.choices[0].message.content or "").strip()

    def _gen_claude(self, prompt: str, system: str) -> str:
        if self._claude_client is None:
            return ""
        kwargs: dict[str, Any] = {
            "model":      self.claude_model,
            "max_tokens": self.max_tokens,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = self._claude_client.messages.create(**kwargs)
        return str(response.content[0].text or "").strip()

    def _gen_gemini(self, prompt: str, system: str) -> str:
        """Generate text using Google Gemini."""
        if self._gemini_client is None:
            return ""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            response = self._gemini_client.generate_content(full_prompt)
            return str(response.text or "").strip()
        except Exception as exc:
            logger.error(f"[ModelInference] Gemini generation error: {exc}")
            return ""

    def _gen_hf(self, prompt: str, system: str) -> str:
        if self._hf_pipeline is None:
            return ""
        full = f"{system}\n\n{prompt}" if system else prompt
        outputs: list[Any] = self._hf_pipeline(full)
        if not outputs or not isinstance(outputs, list):
            return ""
        first = outputs[0]
        if isinstance(first, dict):
            raw  = first.get("generated_text", "")
            text = str(raw).strip()
            if text.startswith(full):
                text = text[len(full):].strip()
            return text
        return ""

    def _gen_ollama(self, prompt: str, system: str) -> str:
        """
        Call a local Ollama server.

        Strategy:
          1. Try the /api/chat endpoint (works with all modern Ollama versions).
          2. Fall back to /api/generate if /api/chat returns a non-200 status.

        Common fix for 404 errors
        -------------------------
        A 404 from /api/generate almost always means the model is not pulled.
        Run:
            ollama pull <model_name>
        then restart the app.

        Popular models: llama3, mistral, phi3, gemma2, qwen2, codellama
        Browse more at: https://ollama.ai/library
        """
        import requests                            # type: ignore

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        # ── attempt 1: /api/chat (preferred — supported since Ollama 0.1.14) ──
        chat_payload: dict[str, Any] = {
            "model":  self.ollama_model,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "messages": [
                *(
                    [{"role": "system", "content": system}] if system else []
                ),
                {"role": "user", "content": prompt},
            ],
        }
        try:
            r = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=chat_payload, timeout=120,
            )
            if r.status_code == 200:
                data = r.json()
                # /api/chat returns {"message": {"role": "assistant", "content": "..."}}
                msg = data.get("message", {})
                text = msg.get("content", "") if isinstance(msg, dict) else ""
                if text:
                    return str(text).strip()
            elif r.status_code == 404:
                # Model not pulled — give an actionable error and fall through
                logger.error(
                    f"[ModelInference] Ollama 404 on /api/chat — "
                    f"model '{self.ollama_model}' is not pulled.\n"
                    f"  Fix:  ollama pull {self.ollama_model}\n"
                    f"  Then restart the app."
                )
                return ""
            else:
                logger.warning(
                    f"[ModelInference] /api/chat returned {r.status_code}, "
                    f"falling back to /api/generate"
                )
        except requests.exceptions.ConnectionError:
            logger.error(
                f"[ModelInference] Cannot connect to Ollama at {self.ollama_base_url}.\n"
                "  Make sure Ollama is running:  ollama serve\n"
                f"  And the model is pulled:      ollama pull {self.ollama_model}"
            )
            return ""
        except Exception as exc:
            logger.warning(f"[ModelInference] /api/chat error: {exc} — trying /api/generate")

        # ── attempt 2: /api/generate (legacy, kept for older Ollama versions) ──
        gen_payload: dict[str, Any] = {
            "model":  self.ollama_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            r2 = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=gen_payload, timeout=120,
            )
            if r2.status_code == 200:
                return str(r2.json().get("response", "")).strip()
            elif r2.status_code == 404:
                logger.error(
                    f"[ModelInference] Ollama 404 on /api/generate — "
                    f"model '{self.ollama_model}' is not pulled.\n"
                    f"  Fix:  ollama pull {self.ollama_model}"
                )
            else:
                logger.error(
                    f"[ModelInference] /api/generate returned HTTP {r2.status_code}: {r2.text[:200]}"
                )
        except requests.exceptions.ConnectionError:
            logger.error(
                f"[ModelInference] Cannot connect to Ollama at {self.ollama_base_url}.\n"
                "  Make sure Ollama is running:  ollama serve"
            )
        except Exception as exc:
            logger.error(f"[ModelInference] Ollama /api/generate error: {exc}")

        return ""

    # ──────────────────────────────────────────────────────
    # JSON parser helper
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _parse_json(text: str) -> Any:
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
        return None

    # ══════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════

    # ── 1. Skill extraction ───────────────────────────────
    def extract_skills(self, text: str) -> list[str]:
        """Extract all skills from free-form text (resume or JD)."""
        system = (
            "You are an expert technical recruiter and NLP system. "
            "Extract all professional skills from the text. "
            "Return ONLY a valid JSON array of skill strings. "
            "No explanations, no markdown, no commentary."
        )
        prompt = f"Extract every technical skill, tool, framework, language, and soft skill:\n\n{text[:4000]}"
        raw = self._generate(prompt, system)
        if raw:
            parsed = self._parse_json(raw)
            if isinstance(parsed, list):
                return self._clean_skills(parsed)
        return self._rule_extract(text)

    def _clean_skills(self, raw: list[Any]) -> list[str]:
        seen, result = set(), []
        for item in raw:
            s = str(item).strip().title()
            if s and s.lower() not in seen and 2 < len(s) < 60:
                seen.add(s.lower())
                result.append(s)
        return result

    def _rule_extract(self, text: str) -> list[str]:
        tl = text.lower()
        return [s for s in SKILL_TAXONOMY
                if re.search(r"\b" + re.escape(s.lower()) + r"\b", tl)]

    # ── 2. Roadmap generation ─────────────────────────────
    def generate_roadmap(
        self,
        skill:         str,
        current_level: str = "beginner",
        target_level:  str = "advanced",
        weeks:         int = 8,
    ) -> dict[str, Any]:
        """Generate a detailed week-by-week learning roadmap for a skill."""
        prereqs = SKILL_PREREQUISITES.get(skill, [])
        system  = (
            "You are a senior Learning & Development specialist. "
            "Create detailed, actionable learning roadmaps. "
            "Return ONLY valid JSON — no markdown, no commentary."
        )
        prompt = f"""
Create a {weeks}-week learning roadmap for "{skill}".
Current level: {current_level}
Target level:  {target_level}
Prerequisites already known: {prereqs}

Return exactly this JSON schema:
{{
  "skill": "{skill}",
  "current_level": "{current_level}",
  "target_level": "{target_level}",
  "duration_weeks": {weeks},
  "prerequisites": ["list"],
  "weekly_plan": [
    {{
      "week": 1,
      "topics": ["topic1", "topic2"],
      "resources": ["https://..."],
      "project": "mini project description",
      "hours": 10
    }}
  ],
  "estimated_hours": 80,
  "priority": "high",
  "key_concepts": ["concept1", "concept2"],
  "career_impact": "brief statement"
}}
"""
        raw = self._generate(prompt, system)
        if raw:
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and "skill" in parsed:
                return parsed
        return self._fallback_roadmap(skill, current_level, target_level, weeks, prereqs)

    @staticmethod
    def _fallback_roadmap(
        skill: str, cur: str, tgt: str, weeks: int, prereqs: list[str],
    ) -> dict[str, Any]:
        phases = ["Foundations", "Core Concepts", "Intermediate", "Advanced", "Projects"]
        plan = [
            {
                "week":      w,
                "topics":    [f"{skill} — {phases[min((w-1)*len(phases)//weeks, len(phases)-1)]}"],
                "resources": [f"https://www.google.com/search?q=learn+{skill.replace(' ', '+')}"],
                "project":   f"{skill} practice project — week {w}",
                "hours":     10,
            }
            for w in range(1, weeks + 1)
        ]
        return {
            "skill": skill, "current_level": cur, "target_level": tgt,
            "duration_weeks": weeks, "prerequisites": prereqs,
            "weekly_plan": plan, "estimated_hours": weeks * 10,
            "priority": "high", "key_concepts": [],
            "career_impact": f"Proficiency in {skill} is highly valued.",
        }

    # ── 3. Experience level detection ─────────────────────
    def detect_experience_level(self, resume_text: str, skill: str) -> str:
        """Return Beginner | Intermediate | Advanced | Expert."""
        system = (
            "You are a technical recruiter assessing candidate skill levels. "
            "Return ONLY one word from: Beginner, Intermediate, Advanced, Expert."
        )
        prompt = (
            f"Based on the resume below, what is the candidate's level with '{skill}'?\n\n"
            f"Resume:\n{resume_text[:2000]}\n\n"
            "Reply with exactly one word: Beginner, Intermediate, Advanced, or Expert."
        )
        raw = self._generate(prompt, system).strip()
        for lvl in ("Expert", "Advanced", "Intermediate", "Beginner"):
            if lvl.lower() in raw.lower():
                return lvl
        return self._heuristic_level(resume_text, skill)

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

    # ── 4. Learning resources ─────────────────────────────
    def generate_learning_roadmap(
        self,
        missing: list[Any],
        weak:    list[Any],
        jd_text: str = "",
    ) -> dict[str, dict[str, Any]]:
        """Generate curated learning resources for missing/weak skills."""
        all_s = [
            s.get("skill", str(s)) if isinstance(s, dict) else str(s)
            for s in (missing + weak)
        ]
        if not all_s:
            return {}

        system = "You are an expert learning coach. Return ONLY valid JSON."
        prompt = f"""
For each skill in this list: {all_s[:10]}
Job context: {jd_text[:400]}

Return a JSON object where each key is a skill name and value is:
{{
  "subtopics": ["topic1", "topic2", "topic3"],
  "resources": ["https://...", "https://..."],
  "projects":  ["project idea 1", "project idea 2"],
  "estimated_hours": 30
}}
"""
        raw = self._generate(prompt, system)
        if raw:
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict):
                return parsed

        return {
            s: {
                "subtopics":       ["Fundamentals", "Intermediate", f"Advanced {s}"],
                "resources":       [f"https://www.google.com/search?q=learn+{s.replace(' ', '+')}"],
                "projects":        [f"Build a small {s} project"],
                "estimated_hours": 30,
            }
            for s in all_s
        }

    # ── 5. Resume suggestions ─────────────────────────────
    def generate_resume_suggestions(
        self, resume_text: str, jd_text: str,
    ) -> list[str]:
        """Return actionable resume improvement suggestions."""
        system = (
            "You are a professional resume coach and technical recruiter. "
            "Return ONLY a JSON array of concise, actionable suggestion strings."
        )
        prompt = (
            "Give exactly 6 specific resume improvement suggestions "
            "based on the resume vs the job description.\n\n"
            f"RESUME (first 2000 chars):\n{resume_text[:2000]}\n\n"
            f"JOB DESCRIPTION (first 1000 chars):\n{jd_text[:1000]}"
        )
        raw = self._generate(prompt, system)
        if raw:
            parsed = self._parse_json(raw)
            if isinstance(parsed, list):
                return [str(s) for s in parsed if s]
        return [
            "Quantify achievements with specific metrics (%, $, time saved).",
            "Mirror keywords from the job description in your skills section.",
            "Add a concise professional summary aligned to the target role.",
            "List your most impactful projects with GitHub / live demo links.",
            "Highlight leadership and cross-team collaboration examples.",
            "Tailor your skills section to match the exact tools in the JD.",
        ]

    # ── 6. Full pipeline ──────────────────────────────────
    def analyze(self, resume_text: str, jd_text: str) -> dict[str, Any]:
        """Run the full analysis pipeline in one call."""
        resume_skills = self.extract_skills(resume_text)
        jd_skills     = self.extract_skills(jd_text)
        resume_set    = {s.lower() for s in resume_skills}
        missing       = [{"skill": s} for s in jd_skills if s.lower() not in resume_set]
        roadmaps      = {s["skill"]: self.generate_roadmap(s["skill"]) for s in missing[:5]}
        suggestions   = self.generate_resume_suggestions(resume_text, jd_text)
        return {
            "resume_skills":  resume_skills,
            "jd_skills":      jd_skills,
            "missing_skills": [s["skill"] for s in missing],
            "suggestions":    suggestions,
            "roadmaps":       roadmaps,
        }

    # ── Ollama public helpers ─────────────────────────────
    def list_ollama_models(self) -> list[str]:
        """Return all models currently pulled in the local Ollama server."""
        try:
            import requests                        # type: ignore
            r = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = r.json().get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
        except Exception as exc:
            logger.warning(f"[ModelInference] Could not list Ollama models: {exc}")
            return []

    def ollama_is_running(self) -> bool:
        """Ping the Ollama server to check if it is reachable."""
        return self._ollama_server_ok()

    def pull_ollama_model(self, model_name: Optional[str] = None) -> bool:
        """
        Pull an Ollama model via the /api/pull endpoint (non-streaming).

        Parameters
        ----------
        model_name : str, optional
            Model to pull.  Defaults to self.ollama_model.

        Returns
        -------
        bool
            True if the pull succeeded, False otherwise.

        Example
        -------
            mi = ModelInference(model_backend='ollama', ollama_model='llama3')
            if mi.backend == 'fallback':
                mi.pull_ollama_model()          # pulls 'llama3'
                mi.backend = 'ollama'           # re-enable after pull
        """
        import requests                            # type: ignore

        target = model_name or self.ollama_model
        logger.info(f"[ModelInference] Pulling Ollama model '{target}' — this may take a while…")
        try:
            r = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json={"name": target, "stream": False},
                timeout=600,                       # large models can take minutes
            )
            if r.status_code == 200:
                logger.info(f"[ModelInference] Successfully pulled '{target}'.")
                # Re-run validation so self.ollama_model and self.backend update
                self.ollama_model = target
                self.backend      = "ollama"
                self._validate_ollama_model()
                return True
            else:
                logger.error(
                    f"[ModelInference] Pull failed — HTTP {r.status_code}: {r.text[:300]}"
                )
                return False
        except Exception as exc:
            logger.error(f"[ModelInference] Pull error: {exc}")
            return False

    def __repr__(self) -> str:
        return (
            f"ModelInference(backend='{self.backend}', "
            f"checkpoint='{self.checkpoint_path}', lora={self.use_lora}, "
            f"gemini_model='{self.gemini_model}', ollama_model='{self.ollama_model}')"
        )