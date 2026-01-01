"""
Entity extraction from patient symptom text.
Extracts: symptoms, duration, severity, body parts.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ExtractedEntities:
    """Container for extracted medical entities."""

    symptoms: List[str]
    duration: Optional[str]
    severity: Optional[str]
    body_parts: List[str]
    negated_symptoms: List[str]


# Symptom patterns mapped to canonical names
SYMPTOM_PATTERNS: dict[str, list[str]] = {
    "chest_pain": [
        r"chest\s*pain",
        r"pain\s*(in|around)?\s*(my|the)?\s*chest",
        r"chest\s*(discomfort|tightness|pressure)",
        r"crushing\s*(chest)?\s*pain",
    ],
    "shortness_of_breath": [
        r"short(ness)?\s*(of)?\s*breath",
        r"difficulty\s*breath",
        r"can'?t\s*breath",
        r"breathing\s*(difficulty|problem|trouble)",
        r"dyspnea",
        r"breathless",
    ],
    "headache": [
        r"head\s*ache",
        r"headache",
        r"pain\s*(in|around)?\s*(my|the)?\s*head",
        r"migraine",
        r"throbbing\s*head",
    ],
    "dizziness": [
        r"dizz(y|iness)",
        r"light\s*head",
        r"vertigo",
        r"feeling\s*faint",
        r"room\s*(is\s*)?spinning",
    ],
    "nausea": [
        r"nausea",
        r"nauseous",
        r"feel(ing)?\s*(like\s*)?(i'?m\s*)?(going\s*to\s*)?throw(ing)?\s*up",
        r"queasy",
        r"sick\s*to\s*(my\s*)?stomach",
    ],
    "vomiting": [
        r"vomit",
        r"threw\s*up",
        r"throwing\s*up",
        r"emesis",
    ],
    "fever": [
        r"fever",
        r"high\s*temperature",
        r"febrile",
        r"feeling\s*hot",
        r"chills?\s*(and\s*fever)?",
    ],
    "cough": [
        r"cough(ing)?",
        r"dry\s*cough",
        r"wet\s*cough",
        r"productive\s*cough",
    ],
    "fatigue": [
        r"fatigu",
        r"tired",
        r"exhausted",
        r"no\s*energy",
        r"weak(ness)?",
        r"lethargy",
    ],
    "abdominal_pain": [
        r"(abdominal|stomach|belly|tummy)\s*pain",
        r"pain\s*(in|around)?\s*(my|the)?\s*(abdomen|stomach|belly)",
        r"stomach\s*ache",
        r"cramp(s|ing)?",
    ],
    "back_pain": [
        r"back\s*pain",
        r"pain\s*(in|around)?\s*(my|the)?\s*back",
        r"lower\s*back",
        r"lumbar\s*pain",
    ],
    "joint_pain": [
        r"joint\s*pain",
        r"(knee|elbow|shoulder|hip|wrist|ankle)\s*pain",
        r"arthr(itis|algia)",
        r"stiff\s*joints?",
    ],
    "rash": [
        r"rash",
        r"skin\s*(irritation|redness|bumps)",
        r"hives",
        r"itchy\s*skin",
        r"eczema",
    ],
    "palpitations": [
        r"palpitation",
        r"heart\s*(racing|pounding|flutter)",
        r"irregular\s*heart",
        r"skipped\s*beat",
    ],
    "sweating": [
        r"sweat(ing)?",
        r"diaphor(esis|etic)",
        r"perspir(ing|ation)",
        r"night\s*sweats?",
    ],
    "numbness": [
        r"numb(ness)?",
        r"tingling",
        r"pins?\s*and\s*needles?",
        r"paresthesia",
    ],
    "blurred_vision": [
        r"blur(red|ry)?\s*vision",
        r"vision\s*(problems?|changes?|loss)",
        r"can'?t\s*see\s*(clearly|well)",
    ],
    "seizure": [
        r"seizure",
        r"convulsion",
        r"fit",
        r"epilep",
    ],
    "confusion": [
        r"confus(ed|ion)",
        r"disoriented",
        r"altered\s*mental",
        r"not\s*making\s*sense",
    ],
    "loss_of_consciousness": [
        r"(lost|loss\s*of)\s*conscious",
        r"passed?\s*out",
        r"faint(ed|ing)?",
        r"syncope",
        r"black(ed)?\s*out",
    ],
}

# Duration patterns
DURATION_PATTERNS: list[tuple[str, str]] = [
    (r"(\d+)\s*min(ute)?s?\s*(ago)?", "minutes"),
    (r"(\d+)\s*hours?\s*(ago)?", "hours"),
    (r"(\d+)\s*days?\s*(ago)?", "days"),
    (r"(\d+)\s*weeks?\s*(ago)?", "weeks"),
    (r"(\d+)\s*months?\s*(ago)?", "months"),
    (r"(a\s*few|couple(\s*of)?)\s*hours?", "few_hours"),
    (r"(a\s*few|couple(\s*of)?)\s*days?", "few_days"),
    (r"(since\s*)?(this\s*)?morning", "today"),
    (r"(since\s*)?(last\s*)?night", "today"),
    (r"(since\s*)?yesterday", "1_day"),
    (r"(for\s*)?(about\s*)?(a\s*)?week", "1_week"),
    (r"suddenly|sudden|abrupt", "sudden_onset"),
    (r"gradual(ly)?", "gradual_onset"),
]

# Severity patterns
SEVERITY_PATTERNS: dict[str, list[str]] = {
    "severe": [
        r"severe",
        r"worst",
        r"extreme",
        r"unbearable",
        r"excruciating",
        r"intense",
        r"10\s*(out\s*of|/)\s*10",
        r"can'?t\s*(stand|bear|tolerate)",
    ],
    "moderate": [
        r"moderate",
        r"medium",
        r"(5|6|7)\s*(out\s*of|/)\s*10",
        r"noticeable",
        r"bothersome",
    ],
    "mild": [
        r"mild",
        r"slight",
        r"minor",
        r"little",
        r"(1|2|3)\s*(out\s*of|/)\s*10",
        r"barely",
    ],
}

# Body part patterns
BODY_PART_PATTERNS: list[str] = [
    r"head",
    r"chest",
    r"(left|right)\s*arm",
    r"(left|right)\s*leg",
    r"back",
    r"neck",
    r"shoulder",
    r"abdomen|stomach|belly",
    r"(left|right)\s*side",
    r"throat",
    r"eye",
    r"ear",
]

# Negation patterns
NEGATION_PATTERNS: list[str] = [
    r"no\s+",
    r"not\s+",
    r"without\s+",
    r"deny\s+",
    r"denies\s+",
    r"doesn'?t\s+have",
    r"don'?t\s+have",
    r"negative\s+for",
]


class EntityExtractor:
    """Extract medical entities from patient symptom text."""

    def __init__(self) -> None:
        """Initialize compiled regex patterns."""
        self._symptom_patterns: dict[str, list[re.Pattern[str]]] = {
            symptom: [re.compile(p, re.IGNORECASE) for p in patterns]
            for symptom, patterns in SYMPTOM_PATTERNS.items()
        }
        self._duration_patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(p, re.IGNORECASE), label)
            for p, label in DURATION_PATTERNS
        ]
        self._severity_patterns: dict[str, list[re.Pattern[str]]] = {
            severity: [re.compile(p, re.IGNORECASE) for p in patterns]
            for severity, patterns in SEVERITY_PATTERNS.items()
        }
        self._body_part_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in BODY_PART_PATTERNS
        ]
        self._negation_pattern = re.compile(
            r"(" + "|".join(NEGATION_PATTERNS) + r")(\w+\s+){0,3}",
            re.IGNORECASE,
        )

    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract all medical entities from text.

        Args:
            text: Patient symptom description

        Returns:
            ExtractedEntities with symptoms, duration, severity, body parts
        """
        # Find negated sections
        negated_sections = self._find_negated_sections(text)

        # Extract symptoms
        symptoms, negated_symptoms = self._extract_symptoms(text, negated_sections)

        # Extract duration
        duration = self._extract_duration(text)

        # Extract severity
        severity = self._extract_severity(text)

        # Extract body parts
        body_parts = self._extract_body_parts(text)

        result = ExtractedEntities(
            symptoms=symptoms,
            duration=duration,
            severity=severity,
            body_parts=body_parts,
            negated_symptoms=negated_symptoms,
        )

        logger.info(
            "entities_extracted",
            symptom_count=len(symptoms),
            negated_count=len(negated_symptoms),
            duration=duration,
            severity=severity,
        )

        return result

    def _find_negated_sections(self, text: str) -> list[tuple[int, int]]:
        """Find text spans that are negated."""
        sections = []
        for match in self._negation_pattern.finditer(text):
            sections.append((match.start(), match.end()))
        return sections

    def _is_negated(self, pos: int, negated_sections: list[tuple[int, int]]) -> bool:
        """Check if a position falls within a negated section."""
        for start, end in negated_sections:
            if start <= pos <= end:
                return True
        return False

    def _extract_symptoms(
        self, text: str, negated_sections: list[tuple[int, int]]
    ) -> tuple[list[str], list[str]]:
        """Extract symptoms, separating negated ones."""
        symptoms = []
        negated = []

        for symptom, patterns in self._symptom_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    if self._is_negated(match.start(), negated_sections):
                        if symptom not in negated:
                            negated.append(symptom)
                    else:
                        if symptom not in symptoms:
                            symptoms.append(symptom)
                    break  # Found match, move to next symptom

        return symptoms, negated

    def _extract_duration(self, text: str) -> Optional[str]:
        """Extract duration/onset information."""
        for pattern, label in self._duration_patterns:
            match = pattern.search(text)
            if match:
                if match.groups() and match.group(1).isdigit():
                    return f"{match.group(1)}_{label}"
                return label
        return None

    def _extract_severity(self, text: str) -> Optional[str]:
        """Extract severity level."""
        for severity, patterns in self._severity_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return severity
        return None

    def _extract_body_parts(self, text: str) -> list[str]:
        """Extract mentioned body parts."""
        body_parts = []
        for pattern in self._body_part_patterns:
            match = pattern.search(text)
            if match:
                part = match.group(0).lower()
                if part not in body_parts:
                    body_parts.append(part)
        return body_parts


# Singleton instance
extractor = EntityExtractor()


def extract_entities(text: str) -> ExtractedEntities:
    """Convenience function to extract entities from text."""
    return extractor.extract(text)
