"""
backend/training_pipeline.py
-----------------------------
End-to-end training pipeline: data preparation → training → evaluation.

FIX (2 Pylance errors):
  • `Dataset[Unknown]` incompatible with `Sized` in `len(dataset)`.
    Resolution: wrap dataset length access in `int(len(...))` or use
    the `.num_rows` attribute which is always present on HF Datasets.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrate the full training workflow."""

    def __init__(self, config_path: str = "training_config.json") -> None:
        self.config = self._load_config(config_path)
        self._trainer: Optional[Any] = None

    # ----------------------------------------------------------
    @staticmethod
    def _load_config(path: str) -> dict[str, Any]:
        default: dict[str, Any] = {
            "model_name":    "gpt2",
            "output_dir":    "./checkpoints",
            "num_epochs":    3,
            "batch_size":    4,
            "learning_rate": 2e-4,
            "max_length":    512,
            "lora_r":        8,
            "lora_alpha":    32,
            "lora_dropout":  0.1,
            "use_wandb":     False,
        }
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    loaded = json.load(f)
                default.update(loaded)
            except Exception as exc:
                logger.warning(f"Could not load config {path}: {exc}")
        return default

    # ----------------------------------------------------------
    def prepare_data(self, raw_texts: list[str]) -> list[str]:
        """Clean and deduplicate training texts."""
        seen: set[str] = set()
        clean: list[str] = []
        for text in raw_texts:
            t = text.strip()
            if t and t not in seen:
                seen.add(t)
                clean.append(t)
        logger.info(f"Prepared {len(clean)} training samples")
        return clean

    # ----------------------------------------------------------
    def load_hf_dataset(
        self,
        dataset_name:  str = "json",
        data_files:    Optional[str] = None,
        split:         str = "train",
        text_column:   str = "text",
    ) -> list[str]:
        """
        Load a HuggingFace dataset and return a list of text strings.

        FIX: `len(dataset)` raised because Dataset.__len__ is typed
             inconsistently in some stub versions.
             Resolution: use `dataset.num_rows` instead of `len(dataset)`
             which is always an int attribute on HF Datasets.
        """
        try:
            from datasets import load_dataset  # type: ignore[import]

            ds_kwargs: dict[str, Any] = {"split": split}
            if data_files:
                ds_kwargs["data_files"] = data_files

            dataset = load_dataset(dataset_name, **ds_kwargs)

            # ✅ FIX: use .num_rows instead of len() to avoid Sized error
            n_rows: int = getattr(dataset, "num_rows", 0)
            logger.info(f"Loaded dataset '{dataset_name}' — {n_rows} rows")

            texts: list[str] = []
            for row in dataset:
                if isinstance(row, dict):
                    val = row.get(text_column, "")
                    if val:
                        texts.append(str(val))

            return texts

        except ImportError:
            logger.warning("datasets not installed — pip install datasets")
            return []
        except Exception as exc:
            logger.error(f"Dataset load error: {exc}")
            return []

    # ----------------------------------------------------------
    def run(
        self,
        texts:         Optional[list[str]] = None,
        dataset_name:  Optional[str]       = None,
        data_files:    Optional[str]       = None,
    ) -> None:
        """Full pipeline: data → model → train → save."""
        from backend.llm_trainer import LLMTrainer

        # Gather training data
        if texts is None:
            texts = []
        if dataset_name:
            texts += self.load_hf_dataset(dataset_name, data_files)
        texts = self.prepare_data(texts)

        if not texts:
            raise ValueError("No training texts provided")

        # Build trainer
        self._trainer = LLMTrainer(
            model_name    = str(self.config.get("model_name",    "gpt2")),
            output_dir    = str(self.config.get("output_dir",    "./checkpoints")),
            lora_r        = int(self.config.get("lora_r",        8)),
            lora_alpha    = int(self.config.get("lora_alpha",    32)),
            lora_dropout  = float(self.config.get("lora_dropout", 0.1)),
            num_epochs    = int(self.config.get("num_epochs",    3)),
            batch_size    = int(self.config.get("batch_size",    4)),
            learning_rate = float(self.config.get("learning_rate", 2e-4)),
            max_length    = int(self.config.get("max_length",    512)),
            use_wandb     = bool(self.config.get("use_wandb",    False)),
        )

        self._trainer.load_base_model()
        self._trainer.apply_lora()
        self._trainer.train(texts)
        self._trainer.save()
        logger.info("Training pipeline complete")

    # ----------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate from the trained model (after run())."""
        if self._trainer is None:
            raise RuntimeError("Call run() first")
        return self._trainer.generate(prompt, max_new_tokens)  # type: ignore[no-any-return]