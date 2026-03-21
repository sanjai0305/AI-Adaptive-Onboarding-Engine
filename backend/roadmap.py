"""
backend/roadmap.py
------------------
Graph-based learning roadmap generator using NetworkX.

FIXES (Pylance errors):
  • `None` assigned to Dict — add default {} guard.
  • `Graph[Unknown]` not assignable to `DiGraph[Node@topological_sort]`
    — use nx.DiGraph explicitly everywhere.
  • `list[Node@topological_sort]` not assignable to `list[str]`
    — cast nodes to str after topological sort.
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# ── Prerequisite graph ──────────────────────────────────────
PREREQUISITES: dict[str, list[str]] = {
    "Machine Learning":   ["Python", "Statistics", "Linear Algebra"],
    "Deep Learning":      ["Machine Learning", "Python"],
    "NLP":                ["Python", "Machine Learning"],
    "Computer Vision":    ["Python", "Deep Learning"],
    "TensorFlow":         ["Python", "Machine Learning"],
    "PyTorch":            ["Python", "Machine Learning"],
    "Kubernetes":         ["Docker", "Linux"],
    "Apache Spark":       ["Python", "SQL"],
    "React":              ["JavaScript", "HTML", "CSS"],
    "Django":             ["Python", "SQL"],
    "FastAPI":            ["Python", "REST APIs"],
    "RAG":                ["Python", "LangChain"],
    "LangChain":          ["Python"],
    "Airflow":            ["Python", "SQL"],
    "dbt":                ["SQL"],
    "Fine-tuning":        ["Python", "PyTorch"],
}

PRIORITY_MAP: dict[str, str] = {
    "Machine Learning": "high", "Deep Learning": "high",
    "Python": "high", "SQL": "high", "Docker": "high",
    "Kubernetes": "high", "TensorFlow": "high", "PyTorch": "high",
    "React": "medium", "Django": "medium", "FastAPI": "medium",
    "NLP": "high", "Computer Vision": "high",
}


class RoadmapEntry(TypedDict):
    skill:         str
    prerequisites: list[str]
    order:         list[str]
    priority:      str


class RoadmapGenerator:
    """Generate prerequisite-aware learning roadmaps."""

    # ----------------------------------------------------------
    @classmethod
    def generate_roadmap(
        cls,
        missing_skills: list[dict[str, Any]],
        weak_skills:    list[dict[str, Any]],
    ) -> dict[str, RoadmapEntry]:
        """
        Build a skill roadmap for missing and weak skills.

        Returns
        -------
        Dict[skill_name → RoadmapEntry]
        """
        # ✅ FIX: default to {} so return type is always dict, never None
        result: dict[str, RoadmapEntry] = {}

        all_skills = [
            s.get("skill", "") for s in (missing_skills + weak_skills)
            if isinstance(s, dict) and s.get("skill")
        ]

        for skill in all_skills:
            prereqs = PREREQUISITES.get(skill, [])
            order   = cls._build_learning_order(skill)
            priority = PRIORITY_MAP.get(skill, "medium")

            result[skill] = RoadmapEntry(
                skill=skill,
                prerequisites=prereqs,
                order=order,
                priority=priority,
            )

        return result

    # ----------------------------------------------------------
    @classmethod
    def _build_learning_order(cls, skill: str) -> list[str]:
        """
        Return a topologically sorted learning order for a skill.

        FIXES:
          • nx.DiGraph used explicitly (not nx.Graph) so topological_sort
            gets a DiGraph — resolves assignability error.
          • Node values after topo sort are cast to str to satisfy
            list[str] return type.
        """
        try:
            import networkx as nx  # type: ignore

            # ✅ FIX: use nx.DiGraph explicitly, not nx.Graph
            graph: nx.DiGraph = nx.DiGraph()

            visited: set[str] = set()
            stack = [skill]

            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                graph.add_node(node)
                for prereq in PREREQUISITES.get(node, []):
                    graph.add_edge(prereq, node)
                    stack.append(prereq)

            # Topological sort → List[Any]; cast each node to str
            # ✅ FIX: list(str(n) for n in ...) satisfies list[str]
            order: list[str] = [
                str(n) for n in nx.topological_sort(graph)  # type: ignore[arg-type]
            ]
            return order

        except ImportError:
            logger.warning("NetworkX not installed — returning flat order")
            prereqs = PREREQUISITES.get(skill, [])
            return prereqs + [skill]

        except Exception as exc:
            logger.error(f"Graph build error for {skill}: {exc}")
            return [skill]

    # ----------------------------------------------------------
    @staticmethod
    def get_priority(skill: str) -> str:
        return PRIORITY_MAP.get(skill, "medium")