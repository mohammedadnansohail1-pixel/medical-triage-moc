"""
Convert DDXPlus evidence codes to natural language text templates.
Based on approach from arXiv 2408.15827.
"""

import json
from pathlib import Path
from typing import Dict, Optional


class EvidenceToTextConverter:
    """Convert DDXPlus evidence codes to natural language."""
    
    def __init__(self, evidences_path: str):
        """Load evidence definitions."""
        with open(evidences_path, 'r') as f:
            self.evidences = json.load(f)
        
        # Build response templates for each evidence
        self.templates = self._build_templates()
        print(f"Loaded {len(self.evidences)} evidence definitions")
        print(f"Built {len(self.templates)} text templates")
    
    def _build_templates(self) -> Dict[str, Dict]:
        """
        Build natural language templates for each evidence.
        Converts questions to first-person statements.
        """
        templates = {}
        
        for code, ev in self.evidences.items():
            if not isinstance(ev, dict):
                continue
            
            question = ev.get("question_en", ev.get("name", ""))
            data_type = ev.get("data_type", "B")
            
            # Convert question to statement
            statement = self._question_to_statement(question)
            
            # Handle value meanings for categorical/multi-choice
            value_meanings = {}
            if "value_meaning" in ev:
                for val_code, meanings in ev["value_meaning"].items():
                    eng_meaning = meanings.get("en", val_code)
                    value_meanings[val_code] = eng_meaning
            
            templates[code] = {
                "statement": statement,
                "question": question,
                "data_type": data_type,
                "value_meanings": value_meanings,
            }
        
        return templates
    
    def _question_to_statement(self, question: str) -> str:
        """Convert a question to a first-person statement."""
        if not question:
            return ""
        
        q = question.strip()
        
        # Remove question mark
        q = q.rstrip("?")
        
        # Common transformations
        transformations = [
            ("Do you have ", "I have "),
            ("Do you feel ", "I feel "),
            ("Are you ", "I am "),
            ("Have you ", "I have "),
            ("Did you ", "I "),
            ("Does the ", "The "),
            ("Is the ", "The "),
            ("Is there ", "There is "),
            ("Are there ", "There are "),
            ("Do you ", "I "),
            ("Can you ", "I can "),
            ("Would you ", "I would "),
        ]
        
        for pattern, replacement in transformations:
            if q.startswith(pattern):
                q = replacement + q[len(pattern):]
                break
        
        # Clean up
        q = q.strip()
        if q and not q.endswith("."):
            q += "."
        
        return q
    
    def convert_evidence(self, evidence_code: str) -> str:
        """
        Convert a single evidence code to natural language.
        
        Args:
            evidence_code: Format "E_XX" or "E_XX_@_V_YY"
            
        Returns:
            Natural language description
        """
        # Parse code and value
        parts = evidence_code.split("_@_")
        base_code = parts[0]
        value_code = parts[1] if len(parts) > 1 else None
        
        if base_code not in self.templates:
            return ""
        
        template = self.templates[base_code]
        statement = template["statement"]
        
        # For categorical/multi-choice, append value meaning
        if value_code and template["value_meanings"]:
            value_meaning = template["value_meanings"].get(value_code, "")
            if value_meaning:
                # Clean up the statement and append value
                statement = statement.rstrip(".")
                statement = f"{statement}: {value_meaning}."
        
        return statement
    
    def convert_evidence_list(self, evidence_codes: list) -> str:
        """
        Convert list of evidence codes to natural language paragraph.
        
        Args:
            evidence_codes: List of evidence codes
            
        Returns:
            Natural language description of symptoms
        """
        statements = []
        seen = set()
        
        for code in evidence_codes:
            # Get base code for deduplication
            base = code.split("_@_")[0]
            
            statement = self.convert_evidence(code)
            if statement and statement not in seen:
                statements.append(statement)
                seen.add(statement)
        
        # Combine into paragraph (limit to avoid too long)
        text = " ".join(statements[:15])
        return text
    
    def convert_patient(
        self,
        evidence_codes: list,
        age: Optional[int] = None,
        sex: Optional[str] = None,
    ) -> str:
        """
        Convert patient data to natural language report.
        
        Args:
            evidence_codes: List of evidence codes
            age: Patient age
            sex: Patient sex (M/F or male/female)
            
        Returns:
            Natural language patient report
        """
        parts = []
        
        # Add demographics
        if age is not None:
            parts.append(f"I am {age} years old.")
        
        if sex:
            sex_text = "male" if sex in ["M", "male"] else "female" if sex in ["F", "female"] else sex
            parts.append(f"I am {sex_text}.")
        
        # Add symptoms
        symptom_text = self.convert_evidence_list(evidence_codes)
        if symptom_text:
            parts.append(symptom_text)
        
        return " ".join(parts)


def test_converter():
    """Test the evidence converter."""
    # Find evidence file
    evidence_path = Path("/home/adnan21/.cache/huggingface/hub/datasets--aai530-group6--ddxplus/snapshots/2ad986acc1ec62fb4a94171acc43f4fdd5bfde53/release_evidences.json")
    
    if not evidence_path.exists():
        evidence_path = Path("data/ddxplus/release_evidences.json")
    
    converter = EvidenceToTextConverter(str(evidence_path))
    
    # Test with sample evidence codes
    print("\n" + "="*60)
    print("SAMPLE CONVERSIONS")
    print("="*60)
    
    test_codes = [
        "E_91",  # Fever
        "E_53",  # Chest pain
        "E_54_@_V_161",  # Pain characteristic
        "E_55_@_V_89",  # Pain location
        "E_77",  # Cough
        "E_201",  # Heartburn
    ]
    
    for code in test_codes:
        text = converter.convert_evidence(code)
        print(f"  {code:20} -> {text}")
    
    # Test full patient conversion
    print("\n" + "="*60)
    print("FULL PATIENT REPORT")
    print("="*60)
    
    sample_evidences = [
        "E_53", "E_54_@_V_161", "E_54_@_V_183", 
        "E_55_@_V_89", "E_91", "E_77", "E_201"
    ]
    
    report = converter.convert_patient(
        evidence_codes=sample_evidences,
        age=45,
        sex="M"
    )
    print(f"\n{report}")
    
    return converter


if __name__ == "__main__":
    test_converter()
