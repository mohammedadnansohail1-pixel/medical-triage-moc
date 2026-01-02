"""
SapBERT Entity Linker - Maps patient symptoms to DDXPlus evidence codes.
Uses embedding similarity - NO hardcoded dictionaries.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import structlog

logger = structlog.get_logger(__name__)


class SapBERTLinker:
    """Links patient language to DDXPlus evidence codes using SapBERT embeddings."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self._loaded = False
        
        # DDXPlus evidence embeddings cache
        self.evidence_embeddings: Optional[torch.Tensor] = None
        self.evidence_codes: List[str] = []
        self.evidence_names: List[str] = []

    def load(self) -> None:
        """Load SapBERT model to GPU/CPU."""
        if self._loaded:
            return

        logger.info("sapbert_loading", model=self.model_name, device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info("sapbert_loaded", device=self.device)

    def unload(self) -> None:
        """Unload model from GPU to free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.evidence_embeddings is not None:
            del self.evidence_embeddings
            self.evidence_embeddings = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("sapbert_unloaded")

    def _get_embedding(self, texts: List[str]) -> torch.Tensor:
        """Get SapBERT embeddings for list of texts."""
        if not self._loaded:
            self.load()

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings

    def build_evidence_index(self, evidences_path: Path) -> None:
        """
        Build embedding index from DDXPlus evidence definitions.
        
        Args:
            evidences_path: Path to release_evidences.json
        """
        if not self._loaded:
            self.load()

        logger.info("building_evidence_index", path=str(evidences_path))

        with open(evidences_path) as f:
            evidences = json.load(f)

        self.evidence_codes = []
        self.evidence_names = []

        for code, info in evidences.items():
            # Get English question/name
            name = info.get("question_en", info.get("name", code))
            # Clean up question format to statement
            name = name.replace("Do you have ", "").replace("?", "")
            name = name.replace("Are you ", "").replace("Did you ", "")
            
            self.evidence_codes.append(code)
            self.evidence_names.append(name)

        # Batch compute embeddings
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(self.evidence_names), batch_size):
            batch = self.evidence_names[i:i + batch_size]
            embeddings = self._get_embedding(batch)
            all_embeddings.append(embeddings)

        self.evidence_embeddings = torch.cat(all_embeddings, dim=0)
        
        logger.info(
            "evidence_index_built",
            num_evidences=len(self.evidence_codes),
            embedding_shape=list(self.evidence_embeddings.shape),
        )

    def link_symptoms(
        self,
        symptoms: List[str],
        top_k: int = 3,
        threshold: float = 0.5,
    ) -> List[Tuple[str, str, float]]:
        """
        Link patient symptoms to DDXPlus evidence codes.
        
        Args:
            symptoms: List of symptom strings from patient
            top_k: Number of top matches per symptom
            threshold: Minimum similarity score
            
        Returns:
            List of (symptom, evidence_code, similarity_score) tuples
        """
        if self.evidence_embeddings is None:
            raise RuntimeError("Evidence index not built. Call build_evidence_index first.")

        if not symptoms:
            return []

        # Get embeddings for input symptoms
        symptom_embeddings = self._get_embedding(symptoms)

        # Compute cosine similarity against all evidences
        # symptom_embeddings: (num_symptoms, hidden_dim)
        # evidence_embeddings: (num_evidences, hidden_dim)
        similarities = F.cosine_similarity(
            symptom_embeddings.unsqueeze(1),  # (num_symptoms, 1, hidden_dim)
            self.evidence_embeddings.unsqueeze(0),  # (1, num_evidences, hidden_dim)
            dim=2,
        )  # (num_symptoms, num_evidences)

        results = []
        for i, symptom in enumerate(symptoms):
            # Get top-k matches for this symptom
            scores, indices = torch.topk(similarities[i], k=min(top_k, len(self.evidence_codes)))
            
            for score, idx in zip(scores, indices):
                if score.item() >= threshold:
                    results.append((
                        symptom,
                        self.evidence_codes[idx.item()],
                        score.item(),
                    ))

        logger.info(
            "symptoms_linked",
            num_input=len(symptoms),
            num_matches=len(results),
        )

        return results

    def get_matched_codes(
        self,
        symptoms: List[str],
        threshold: float = 0.5,
    ) -> List[str]:
        """
        Get unique evidence codes for symptoms.
        
        Args:
            symptoms: List of symptom strings
            threshold: Minimum similarity score
            
        Returns:
            List of unique DDXPlus evidence codes
        """
        matches = self.link_symptoms(symptoms, top_k=1, threshold=threshold)
        codes = list(set(code for _, code, _ in matches))
        return codes


# Singleton instance
_linker: Optional[SapBERTLinker] = None


def get_sapbert_linker() -> SapBERTLinker:
    """Get or create singleton SapBERT linker."""
    global _linker
    if _linker is None:
        _linker = SapBERTLinker()
    return _linker
