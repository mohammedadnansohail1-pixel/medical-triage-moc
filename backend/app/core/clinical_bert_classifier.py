"""
ClinicalBERT Classifier - Routes symptoms to medical specialties.
Fine-tuned on DDXPlus pathology -> specialty mapping.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import structlog

logger = structlog.get_logger(__name__)


class ClinicalBERTClassifier(nn.Module):
    """ClinicalBERT with classification head for specialty routing."""

    def __init__(self, num_classes: int, model_name: str = "emilyalsentzer/Bio_ClinicalBERT") -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class SpecialtyClassifier:
    """Wrapper for ClinicalBERT specialty classification."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[ClinicalBERTClassifier] = None
        self.specialties: List[str] = []
        self._loaded = False

    def load(self, model_path: Optional[Path] = None, specialties: Optional[List[str]] = None) -> None:
        """Load model - either pretrained or from checkpoint."""
        if self._loaded:
            return

        logger.info("clinical_bert_loading", device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        if model_path and model_path.exists():
            # Load fine-tuned model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.specialties = checkpoint["specialties"]
            self.model = ClinicalBERTClassifier(num_classes=len(self.specialties))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("clinical_bert_loaded_checkpoint", path=str(model_path))
        else:
            # Initialize new model
            if specialties is None:
                raise ValueError("Must provide specialties for new model")
            self.specialties = specialties
            self.model = ClinicalBERTClassifier(num_classes=len(specialties))
            logger.info("clinical_bert_initialized_new", num_classes=len(specialties))

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def unload(self) -> None:
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("clinical_bert_unloaded")

    def predict(self, texts: List[str], top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """
        Predict specialties for input texts.
        
        Returns list of [(specialty, confidence), ...] for each input.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=-1)

        results = []
        for i in range(len(texts)):
            scores, indices = torch.topk(probs[i], k=min(top_k, len(self.specialties)))
            predictions = [
                (self.specialties[idx.item()], score.item())
                for score, idx in zip(scores, indices)
            ]
            results.append(predictions)

        return results

    def save(self, path: Path) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "specialties": self.specialties,
        }, path)
        logger.info("clinical_bert_saved", path=str(path))


_classifier: Optional[SpecialtyClassifier] = None

def get_specialty_classifier() -> SpecialtyClassifier:
    global _classifier
    if _classifier is None:
        _classifier = SpecialtyClassifier()
    return _classifier
