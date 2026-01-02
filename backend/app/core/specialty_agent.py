"""
Specialty Agents for Differential Diagnosis.

After Tier 1 routes to a specialty, these agents produce
a ranked differential diagnosis of conditions within that specialty.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


class SpecialtyAgent:
    """
    Base agent for differential diagnosis within a specialty.
    
    Uses Naive Bayes:
      P(condition | symptoms) ∝ P(condition) × Π P(symptom | condition)
    """
    
    def __init__(self, specialty: str, model_data: Dict):
        """
        Initialize agent for a specific specialty.
        
        Args:
            specialty: One of cardiology, pulmonology, etc.
            model_data: Pre-loaded condition_model.json data
        """
        self.specialty = specialty
        
        if specialty not in model_data["specialties"]:
            raise ValueError(f"Unknown specialty: {specialty}")
        
        spec_data = model_data["specialties"][specialty]
        self.conditions = spec_data["conditions"]
        self.priors = spec_data["priors"]
        self.symptom_probs = spec_data["symptom_probs"]
        self.symptom_questions = model_data.get("symptom_questions", {})
    
    def diagnose(
        self,
        symptom_codes: List[str],
        age: Optional[int] = None,
        sex: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Generate differential diagnosis for given symptoms.
        
        Args:
            symptom_codes: List of E_XX evidence codes
            age: Patient age (for future use)
            sex: Patient sex (for future use)
            top_k: Number of conditions to return
            
        Returns:
            List of (condition_name, probability) sorted by probability desc
        """
        symptom_set = set(symptom_codes)
        
        # Calculate log posterior for each condition
        log_posteriors = {}
        
        for condition in self.conditions:
            # Start with log prior
            prior = self.priors.get(condition, 1e-6)
            log_post = math.log(max(prior, 1e-10))
            
            # Add log likelihood for each symptom
            cond_probs = self.symptom_probs.get(condition, {})
            
            for symptom in symptom_set:
                # P(symptom | condition), with smoothing
                p_sym = cond_probs.get(symptom, 0.01)
                p_sym = max(0.001, min(0.999, p_sym))  # Clamp
                log_post += math.log(p_sym)
            
            log_posteriors[condition] = log_post
        
        # Convert to probabilities via softmax
        max_log = max(log_posteriors.values())
        exp_posts = {c: math.exp(lp - max_log) for c, lp in log_posteriors.items()}
        total = sum(exp_posts.values())
        
        posteriors = {c: exp_posts[c] / total for c in exp_posts}
        
        # Sort and return top-k
        ranked = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    def get_condition_symptoms(self, condition: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get most indicative symptoms for a condition."""
        if condition not in self.symptom_probs:
            return []
        
        probs = self.symptom_probs[condition]
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class SpecialtyAgentManager:
    """
    Manages all specialty agents.
    
    Loads model once, creates agents on demand.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize manager.
        
        Args:
            model_path: Path to condition_model.json
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent.parent / "data" / "ddxplus" / "condition_model.json"
        
        self.model_path = Path(model_path)
        self._model_data: Optional[Dict] = None
        self._agents: Dict[str, SpecialtyAgent] = {}
    
    def _load_model(self) -> None:
        """Load model data if not already loaded."""
        if self._model_data is not None:
            return
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with open(self.model_path) as f:
            self._model_data = json.load(f)
        
        logger.info("specialty_model_loaded", 
                   path=str(self.model_path),
                   specialties=list(self._model_data["specialties"].keys()))
    
    def get_agent(self, specialty: str) -> SpecialtyAgent:
        """
        Get or create agent for a specialty.
        
        Args:
            specialty: Specialty name
            
        Returns:
            SpecialtyAgent for that specialty
        """
        self._load_model()
        
        if specialty not in self._agents:
            self._agents[specialty] = SpecialtyAgent(specialty, self._model_data)
            logger.info("specialty_agent_created", specialty=specialty)
        
        return self._agents[specialty]
    
    def diagnose(
        self,
        specialty: str,
        symptom_codes: List[str],
        age: Optional[int] = None,
        sex: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Convenience method: get agent and diagnose in one call.
        """
        agent = self.get_agent(specialty)
        return agent.diagnose(symptom_codes, age, sex, top_k)
    
    @property
    def available_specialties(self) -> List[str]:
        """List of available specialties."""
        self._load_model()
        return list(self._model_data["specialties"].keys())


# Singleton instance
_manager: Optional[SpecialtyAgentManager] = None


def get_specialty_manager(model_path: Optional[Path] = None) -> SpecialtyAgentManager:
    """Get or create singleton manager."""
    global _manager
    if _manager is None:
        _manager = SpecialtyAgentManager(model_path)
    return _manager
