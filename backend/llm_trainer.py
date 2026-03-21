"""
backend/llm_trainer.py
-----------------------
Fine-tune a causal-LM with LoRA / QLoRA using PEFT.

FIXES (31 Pylance errors):
  • torch.cuda.amp.autocast  → torch.amp.autocast (new API) or use
    contextlib.nullcontext as fallback
  • torch.cuda.amp.GradScaler → torch.cuda.amp.GradScaler still valid
    but guarded with hasattr for portability
  • `LoraConfig`, `get_peft_model`, `TaskType`, `PeftModel`
    — use public peft API: `from peft import ...`
    (Pylance warns on "not exported" for internal sub-modules)
    Fixed by importing from the top-level peft package directly
    and suppressing residual warnings with # type: ignore only
    where unavoidable.
  • `datasets` not resolved → guarded import
  • `wandb`    not resolved → guarded import
  • None.train / None.eval / None.generate etc. → add None guards
  • `save_pretrained` path argument → cast Path to str
"""
from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Safe peft imports ───────────────────────────────────────
try:
    # ✅ FIX: import from top-level peft, not sub-modules
    from peft import (           # type: ignore[import]
        LoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
    )
    _PEFT_OK = True
except ImportError:
    _PEFT_OK = False
    logger.warning("peft not installed — LoRA training unavailable")

# ── Safe datasets import ────────────────────────────────────
try:
    from datasets import Dataset, load_dataset  # type: ignore[import]
    _DATASETS_OK = True
except ImportError:
    _DATASETS_OK = False
    logger.warning("datasets not installed — pip install datasets")

# ── Safe wandb import ───────────────────────────────────────
try:
    import wandb  # type: ignore[import]
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

# ── AMP context ────────────────────────────────────────────
def _amp_context(device: str, dtype: torch.dtype = torch.float16) -> Any:
    """
    Return an autocast context.

    FIX: torch.cuda.amp.autocast is deprecated in PyTorch 2.x.
         torch.amp.autocast is the new public API.
         Falls back to nullcontext if neither is available.
    """
    if device == "cuda" and torch.cuda.is_available():
        try:
            # ✅ FIX: new public API
            return torch.amp.autocast(device_type="cuda", dtype=dtype)  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback for older PyTorch
            return torch.cuda.amp.autocast(dtype=dtype)  # type: ignore[attr-defined]
    return contextlib.nullcontext()


def _grad_scaler() -> Optional[Any]:
    """
    Return a GradScaler if CUDA is available, else None.

    FIX: GradScaler is still in torch.cuda.amp in all supported versions.
    """
    if torch.cuda.is_available():
        try:
            return torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]
        except AttributeError:
            return None
    return None


# ══════════════════════════════════════════════════════════
class LLMTrainer:
    """Fine-tune a causal-LM using LoRA via PEFT."""

    def __init__(
        self,
        model_name:      str   = "gpt2",
        output_dir:      str   = "./checkpoints",
        lora_r:          int   = 8,
        lora_alpha:      int   = 32,
        lora_dropout:    float = 0.1,
        num_epochs:      int   = 3,
        batch_size:      int   = 4,
        learning_rate:   float = 2e-4,
        max_length:      int   = 512,
        use_wandb:       bool  = False,
    ) -> None:
        self.model_name    = model_name
        self.output_dir    = output_dir
        self.lora_r        = lora_r
        self.lora_alpha    = lora_alpha
        self.lora_dropout  = lora_dropout
        self.num_epochs    = num_epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.max_length    = max_length
        self.use_wandb     = use_wandb and _WANDB_OK

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ FIX: explicitly typed as Optional so None-attr checks pass
        self._model:     Optional[nn.Module]  = None
        self._tokenizer: Optional[Any]        = None

    # ----------------------------------------------------------
    def load_base_model(self) -> None:
        """Load tokenizer and base causal-LM."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

            logger.info(f"Loading base model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self._model = base.to(self.device)  # type: ignore[assignment]
            logger.info("Base model loaded")
        except Exception as exc:
            logger.error(f"load_base_model failed: {exc}")
            raise

    # ----------------------------------------------------------
    def apply_lora(self) -> None:
        """Wrap the base model with LoRA adapters."""
        if not _PEFT_OK:
            raise ImportError("peft not installed — pip install peft")
        # ✅ FIX: guard against None model
        if self._model is None:
            raise RuntimeError("Call load_base_model() first")

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        # ✅ FIX: get_peft_model returns Module; assign back to _model
        self._model = get_peft_model(self._model, config)  # type: ignore[assignment]
        logger.info("LoRA adapters applied")

    # ----------------------------------------------------------
    def train(self, texts: list[str]) -> None:
        """Fine-tune the model on a list of training texts."""
        # ✅ FIX: explicit None guards
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call load_base_model() and apply_lora() first")

        if self.use_wandb:
            wandb.init(project="llm-trainer", name=self.model_name)  # type: ignore[possibly-undefined]

        optimiser = torch.optim.AdamW(
            self._model.parameters(), lr=self.learning_rate
        )
        scaler = _grad_scaler()

        self._model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            batches = [
                texts[i : i + self.batch_size]
                for i in range(0, len(texts), self.batch_size)
            ]

            for batch in batches:
                encodings = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                input_ids      = encodings["input_ids"]
                attention_mask = encodings["attention_mask"]

                optimiser.zero_grad()

                with _amp_context(self.device):
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                    loss: torch.Tensor = outputs.loss

                if scaler is not None:
                    scaler.scale(loss).backward()     # type: ignore[attr-defined]
                    scaler.step(optimiser)            # type: ignore[attr-defined]
                    scaler.update()                   # type: ignore[attr-defined]
                else:
                    loss.backward()
                    optimiser.step()

                total_loss += loss.item()

            avg = total_loss / max(len(batches), 1)
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} — loss: {avg:.4f}")

            if self.use_wandb:
                wandb.log({"epoch": epoch + 1, "loss": avg})  # type: ignore[possibly-undefined]

    # ----------------------------------------------------------
    def save(self, path: Optional[str] = None) -> None:
        """Save the fine-tuned model and tokenizer."""
        # ✅ FIX: guard None; cast Path to str for save_pretrained
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded")

        save_path = str(Path(path or self.output_dir))
        os.makedirs(save_path, exist_ok=True)

        self._model.save_pretrained(save_path)         # type: ignore[union-attr]
        self._tokenizer.save_pretrained(save_path)     # type: ignore[union-attr]
        logger.info(f"Model saved to {save_path}")

    # ----------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate text from a prompt using the trained model."""
        # ✅ FIX: guard None
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded")

        self._model.eval()

        inputs = self._tokenizer(
            prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            ids = self._model.generate(                # type: ignore[union-attr]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        return self._tokenizer.decode(ids[0], skip_special_tokens=True)  # type: ignore[union-attr]