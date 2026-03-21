# train.py
"""
==============================================================
MAIN TRAINING SCRIPT
==============================================================
Run this script to train LLM models
"""

import argparse
import logging
from pathlib import Path
import json
from backend.training_pipeline import TrainingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Train LLM models for Adaptive Onboarding Engine'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-uncased',
        help='Base model name from Hugging Face'
    )
    
    parser.add_argument(
        '--decoder-model',
        type=str,
        default='gpt2',
        help='Decoder model for sequence generation'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--use-lora',
        action='store_true',
        default=True,
        help='Use LoRA for parameter-efficient training'
    )
    
    parser.add_argument(
        '--use-qlora',
        action='store_true',
        default=False,
        help='Use QLoRA for quantized training'
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        default=False,
        help='Log to Weights & Biases'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./model_checkpoints',
        help='Output directory for checkpoints'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file'
    )
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = {
            'model_name': args.model,
            'decoder_model': args.decoder_model,
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'use_lora': args.use_lora,
            'use_qlora': args.use_qlora,
            'use_wandb': args.use_wandb,
            'checkpoint_dir': args.output_dir
        }
    
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60 + "\n")
    
    # Create pipeline and run
    pipeline = TrainingPipeline(config)
    pipeline.run_complete_pipeline()
    
    logger.info("\n✅ Training complete!")


if __name__ == '__main__':
    main()