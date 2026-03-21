# backend/tensorflow_llm_trainer.py
"""
==============================================================
LLM TRAINING MODULE - TensorFlow Implementation
==============================================================
Features:
- TensorFlow/Keras LLM fine-tuning
- KerasNLP integration
- Mixed precision training
- DistributedDataParallel support
- Custom layers and loss functions
- Model evaluation metrics

Author: AI Forge Squad
Date: March 2026
Version: 1.0
==============================================================
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime
import json

# KerasNLP for transformer models
import keras_nlp
from transformers import TFAutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ====== TENSORFLOW LLM TRAINER ======
class TensorFlowLLMTrainer:
    """
    TensorFlow-based LLM trainer with mixed precision and distributed support
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        output_dir: str = "./tf_model_checkpoints",
        use_mixed_precision: bool = True,
        distribute_strategy: str = "mirrored"  # mirrored, tpu, or none
    ):
        """
        Initialize TensorFlow LLM Trainer
        
        Args:
            model_name: Hugging Face model ID
            output_dir: Directory for checkpoints
            use_mixed_precision: Use mixed precision (float16)
            distribute_strategy: Distribution strategy
        """
        
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")
        
        # Distribution strategy
        if distribute_strategy == "mirrored":
            self.strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Using MirroredStrategy with {self.strategy.num_replicas_in_sync} replicas")
        elif distribute_strategy == "tpu":
            self.strategy = tf.distribute.TPUStrategy()
            logger.info("Using TPU Strategy")
        else:
            self.strategy = tf.distribute.get_strategy()
            logger.info("Using default distribution strategy")
        
        self.model = None
        self.tokenizer = None
        self.history = {'loss': [], 'val_loss': []}
        
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load with distribution strategy
        with self.strategy.scope():
            self.model = TFAutoModelForCausalLM.from_pretrained(
                self.model_name,
                from_pt=True  # Convert from PyTorch checkpoint
            )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def create_tf_dataset(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from texts
        
        Args:
            texts: List of text samples
            batch_size: Batch size
            max_length: Max token length
            shuffle: Shuffle data
            
        Returns:
            tf.data.Dataset
        """
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': encodings['input_ids']  # Labels are input ids for CLM
        })
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def compile_model(
        self,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01
    ):
        """Compile model with optimizer and loss"""
        
        with self.strategy.scope():
            # Optimizer with weight decay
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
            
            # Loss function
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )
            
            # Metrics
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
            ]
            
            # Compile
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
        
        logger.info("Model compiled successfully")
    
    def train(
        self,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
        max_length: int = 512,
        use_callbacks: bool = True
    ):
        """
        Train the model
        
        Args:
            train_texts: Training text samples
            val_texts: Validation text samples
            num_epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_length: Max sequence length
            use_callbacks: Use TensorBoard, early stopping, etc.
        """
        
        logger.info("Preparing training data...")
        
        # Create datasets
        train_dataset = self.create_tf_dataset(
            train_texts,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=True
        )
        
        val_dataset = None
        if val_texts:
            val_dataset = self.create_tf_dataset(
                val_texts,
                batch_size=batch_size,
                max_length=max_length,
                shuffle=False
            )
        
        # Compile model
        self.compile_model(learning_rate=learning_rate)
        
        # Callbacks
        callbacks = []
        
        if use_callbacks:
            callbacks.extend([
                tf.keras.callbacks.TensorBoard(
                    log_dir=self.output_dir / 'logs',
                    histogram_freq=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.output_dir / 'checkpoint.h5'),
                    save_best_only=True,
                    monitor='val_loss' if val_dataset else 'loss'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if val_dataset else 'loss',
                    patience=3,
                    restore_best_weights=True
                )
            ])
        
        # Train
        logger.info("Starting training...")
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        logger.info("Training completed!")
    
    def evaluate(self, eval_texts: List[str], batch_size: int = 8) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            eval_texts: Evaluation text samples
            batch_size: Batch size
            
        Returns:
            Evaluation metrics
        """
        
        eval_dataset = self.create_tf_dataset(
            eval_texts,
            batch_size=batch_size,
            shuffle=False
        )
        
        results = self.model.evaluate(eval_dataset)
        
        return {
            'loss': results[0],
            'accuracy': results[1]
        }
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using the model
        
        Args:
            prompt: Input prompt
            max_length: Max generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            Generated text
        """
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='tf'
        )
        
        # Generate
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0].numpy(),
            skip_special_tokens=True
        )
        
        return generated_text
    
    def save_model(self, checkpoint_name: str = "final"):
        """Save trained model"""
        
        save_path = self.output_dir / checkpoint_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save history
        with open(save_path / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, checkpoint_dir: str):
        """Load trained model"""
        
        checkpoint_path = Path(checkpoint_dir)
        
        with self.strategy.scope():
            self.model = TFAutoModelForCausalLM.from_pretrained(checkpoint_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        logger.info(f"Model loaded from {checkpoint_path}")