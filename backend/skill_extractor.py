"""
backend/skill_extractor.py
--------------------------
Hybrid skill extractor: rule-based keyword matching + SpaCy NER.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

# ── Skill taxonomy ─────────────────────────────────────────────────────────
SKILL_DB: list[str] = [
    # Languages
    "Python","Java","JavaScript","TypeScript","C++","C#","Go","Rust",
    "Kotlin","Swift","R","Scala","PHP","Ruby","Bash","SQL","HTML","CSS",
    # ML / AI
    "Machine Learning","Deep Learning","NLP","Computer Vision",
    "Reinforcement Learning","Statistical Modeling","Data Analysis",
    "Feature Engineering","Model Evaluation","A/B Testing","LLM",
    "Prompt Engineering","RAG","Fine-tuning","LoRA","Transfer Learning",
    # Frameworks
    "TensorFlow","PyTorch","Keras","Scikit-learn","XGBoost","LightGBM",
    "Hugging Face","Transformers","FastAPI","Django","Flask","React",
    "Next.js","Vue.js","Angular","Node.js","Spring Boot","Express.js",
    "LangChain","LlamaIndex",
    # Data
    "Pandas","NumPy","Matplotlib","Seaborn","Plotly","Tableau","Power BI",
    "PostgreSQL","MySQL","MongoDB","Redis","Elasticsearch","Cassandra",
    "BigQuery","Snowflake","dbt","SQLAlchemy",
    # Cloud & DevOps
    "AWS","Azure","GCP","Docker","Kubernetes","Terraform","CI/CD",
    "Jenkins","GitHub Actions","Linux","Nginx","Ansible",
    # Data Engineering
    "Apache Spark","Apache Kafka","Airflow","ETL","Hadoop","Flink",
    # Tools
    "Git","GitHub","JIRA","Confluence","Jupyter","VS Code","Postman",
    "OpenAI API","Anthropic API","Streamlit","Gradio",
    # Soft skills
    "Communication","Leadership","Team Collaboration","Problem Solving",
    "Project Management","Agile","Scrum","Critical Thinking",
]


class SkillExtractor:
    """Extract skills from text using hybrid methods."""

    @classmethod
    def extract_skills(
        cls,
        text: str,
        method: Literal["rule", "spacy", "hybrid"] = "hybrid",
    ) -> list[str]:
        """
        Extract skill mentions from text.

        Parameters
        ----------
        text   : raw document text
        method : 'rule' | 'spacy' | 'hybrid'

        Returns
        -------
        Deduplicated list of normalised skill strings.
        """
        if not text:
            return []

        if method == "rule":
            return cls._rule_based(text)
        elif method == "spacy":
            return cls._spacy_based(text)
        else:  # hybrid
            rule_skills  = cls._rule_based(text)
            spacy_skills = cls._spacy_based(text)
            return cls._deduplicate(rule_skills + spacy_skills)

    # ----------------------------------------------------------
    @staticmethod
    def _rule_based(text: str) -> list[str]:
        """Keyword matching against SKILL_DB."""
        found: list[str] = []
        text_lower = text.lower()
        for skill in SKILL_DB:
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found.append(skill)
        return found

    @staticmethod
    def _spacy_based(text: str) -> list[str]:
        """NER + noun-chunk extraction using SpaCy."""
        try:
            import spacy  # type: ignore

            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy model not found. Run: python -m spacy download en_core_web_sm")
                return []

            doc = nlp(text[:10_000])  # limit for speed
            candidates: list[str] = []

            # Named entities (ORG, PRODUCT often capture tech names)
            for ent in doc.ents:
                if ent.label_ in ("ORG", "PRODUCT", "GPE"):
                    candidates.append(ent.text.strip())

            # Noun chunks that look like skills
            for chunk in doc.noun_chunks:
                token = chunk.text.strip()
                if 2 < len(token) < 40 and not token[0].isdigit():
                    candidates.append(token)

            return candidates

        except ImportError:
            return []

    @staticmethod
    def _deduplicate(skills: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for s in skills:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                result.append(s)
        return result