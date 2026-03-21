# backend/skill_extraction_model.py
"""
==============================================================
SKILL EXTRACTION MODEL - Fine-tuned for Token Classification
==============================================================
Custom model for extracting skills from resumes and JDs
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# ====== SKILL EXTRACTION MODEL ======
class SkillExtractionModel(nn.Module):
    """
    Token classification model for skill extraction.
    Base model + linear head for token classification.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 3,  # B-SKILL, I-SKILL, O
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1
    ):
        """
        Args:
            model_name: Base model from Hugging Face
            num_labels: Number of token classification labels
            hidden_dropout_prob: Dropout probability
            attention_probs_dropout_prob: Attention dropout
        """
        
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load base model
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout and dense layers
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> Dict:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Token labels (optional)
            
        Returns:
            Dictionary with logits and optional loss
        """
        
        # BERT forward
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Dropout and classification
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        result = {'logits': logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
            result['loss'] = loss
        
        return result
    
    def extract_skills(
        self,
        text: str,
        tokenizer,
        device: str = 'cpu',
        threshold: float = 0.5
    ) -> List[str]:
        """
        Extract skills from text
        
        Args:
            text: Input text
            tokenizer: Hugging Face tokenizer
            device: Device to run on
            threshold: Confidence threshold
            
        Returns:
            List of extracted skills
        """
        
        self.eval()
        self.to(device)
        
        # Tokenize
        encodings = tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(**encodings)
            logits = outputs['logits']
        
        # Get predictions
        predictions = torch.argmax(logits, dim=2)
        confidences = torch.softmax(logits, dim=2)
        
        # Decode predictions
        tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'][0])
        pred_labels = predictions[0].cpu().numpy()
        pred_confidences = confidences[0].cpu().numpy()
        
        # Extract skills (label 1 = B-SKILL, 2 = I-SKILL)
        skills = []
        current_skill = []
        
        for token, label, confidence in zip(tokens, pred_labels, pred_confidences):
            if label in [1, 2] and confidence[label] > threshold:
                if token.startswith('##'):
                    current_skill.append(token[2:])
                else:
                    if current_skill:
                        skills.append(''.join(current_skill))
                    current_skill = [token]
            else:
                if current_skill:
                    skills.append(''.join(current_skill))
                    current_skill = []
        
        if current_skill:
            skills.append(''.join(current_skill))
        
        return list(set(skills))


# ====== ROADMAP GENERATION MODEL ======
class RoadmapGenerationModel(nn.Module):
    """
    Sequence-to-sequence model for roadmap generation.
    Generates learning steps given a skill.
    """
    
    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        decoder_name: str = "gpt2",
        hidden_size: int = 256
    ):
        """
        Args:
            encoder_name: Encoder model
            decoder_name: Decoder model
            hidden_size: Hidden size for projection
        """
        
        super().__init__()
        
        # Encoder (understand input skill)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Decoder (generate roadmap steps)
        self.decoder = AutoModel.from_pretrained(decoder_name)
        
        # Projection layer
        self.projection = nn.Linear(
            self.encoder.config.hidden_size,
            self.decoder.config.hidden_size
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Encoder input (skill)
            decoder_input_ids: Decoder input (roadmap prefix)
            
        Returns:
            Logits for generation
        """
        
        # Encode
        encoder_outputs = self.encoder(input_ids=input_ids, return_dict=True)
        encoder_hidden = encoder_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Project
        decoder_hidden = self.projection(encoder_hidden)
        
        # Decode (simplified - in practice use attention mechanism)
        if decoder_input_ids is not None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                return_dict=True
            )
            return decoder_outputs.logits
        
        return None