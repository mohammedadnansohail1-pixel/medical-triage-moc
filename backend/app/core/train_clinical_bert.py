"""
Fine-tune ClinicalBERT on DDXPlus for specialty classification.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import structlog

from app.core.clinical_bert_classifier import ClinicalBERTClassifier

logger = structlog.get_logger(__name__)


class DDXPlusDataset(Dataset):
    """Dataset for DDXPlus pathology -> specialty training."""

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: AutoTokenizer,
        specialty_to_idx: Dict[str, int],
        evidences: Dict,
        max_length: int = 256,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.specialty_to_idx = specialty_to_idx
        self.evidences = evidences
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def _evidence_to_text(self, evidence_list: List[str]) -> str:
        """Convert evidence codes to natural text."""
        texts = []
        for ev in evidence_list:
            # Parse evidence: "E_XX_@_value"
            parts = ev.split("_@_")
            code = parts[0]
            
            if code in self.evidences:
                q = self.evidences[code].get("question_en", code)
                texts.append(q)
        
        return " ".join(texts[:10])  # Limit to first 10 evidences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Build text from evidences
        text = self._evidence_to_text(sample.get("EVIDENCES", []))
        
        # Get specialty label
        pathology = sample.get("PATHOLOGY", "")
        specialty = self._pathology_to_specialty(pathology)
        label = self.specialty_to_idx.get(specialty, 0)

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _pathology_to_specialty(self, pathology: str) -> str:
        """Map DDXPlus pathology to specialty."""
        # This mapping comes from the conditions file
        pathology_lower = pathology.lower()
        
        if any(x in pathology_lower for x in ["heart", "cardiac", "coronary", "angina", "myocard"]):
            return "cardiology"
        elif any(x in pathology_lower for x in ["lung", "pulmon", "pneum", "bronch", "asthma", "copd"]):
            return "pulmonology"
        elif any(x in pathology_lower for x in ["brain", "neuro", "stroke", "seizure", "migraine"]):
            return "neurology"
        elif any(x in pathology_lower for x in ["stomach", "intestin", "gastro", "bowel", "liver", "gerd"]):
            return "gastroenterology"
        elif any(x in pathology_lower for x in ["bone", "joint", "fracture", "arthri", "muscle"]):
            return "orthopedics"
        elif any(x in pathology_lower for x in ["skin", "rash", "derma", "eczema"]):
            return "dermatology"
        elif any(x in pathology_lower for x in ["emergency", "anaphyl", "shock"]):
            return "emergency"
        else:
            return "general_medicine"


def load_ddxplus_data(data_dir: Path) -> Tuple[List[Dict], List[Dict], Dict]:
    """Load DDXPlus train/test splits and evidences."""
    train_path = data_dir / "release_train_patients.json"
    test_path = data_dir / "release_test_patients.json"
    evidences_path = data_dir / "release_evidences.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)
    with open(evidences_path) as f:
        evidences = json.load(f)

    return train_data, test_data, evidences


def train(
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    max_samples: int = None,
) -> None:
    """Fine-tune ClinicalBERT on DDXPlus."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("training_start", device=device, batch_size=batch_size, epochs=epochs)

    # Load data
    train_data, test_data, evidences = load_ddxplus_data(data_dir)
    
    if max_samples:
        train_data = train_data[:max_samples]
        test_data = test_data[:max_samples // 5]

    logger.info("data_loaded", train=len(train_data), test=len(test_data))

    # Define specialties
    specialties = [
        "cardiology", "pulmonology", "neurology", "gastroenterology",
        "orthopedics", "dermatology", "emergency", "general_medicine"
    ]
    specialty_to_idx = {s: i for i, s in enumerate(specialties)}

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = ClinicalBERTClassifier(num_classes=len(specialties))
    model.to(device)

    # Create datasets
    train_dataset = DDXPlusDataset(train_data, tokenizer, specialty_to_idx, evidences)
    test_dataset = DDXPlusDataset(test_data, tokenizer, specialty_to_idx, evidences)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        logger.info("epoch_complete", epoch=epoch+1, loss=avg_loss, accuracy=accuracy)

        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "specialties": specialties,
            }, output_dir / "clinical_bert_classifier.pt")
            logger.info("model_saved", accuracy=accuracy)

    logger.info("training_complete", best_accuracy=best_acc)


if __name__ == "__main__":
    data_dir = Path("/home/adnan21/projects/medical-triage-moc/data/ddxplus")
    output_dir = Path("/home/adnan21/projects/medical-triage-moc/models")
    
    train(data_dir, output_dir, batch_size=8, epochs=3, max_samples=10000)
