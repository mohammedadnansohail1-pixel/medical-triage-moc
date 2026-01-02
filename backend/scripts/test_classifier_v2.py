#!/usr/bin/env python3
"""Test the new classifier module."""

import sys
sys.path.insert(0, "backend")

from app.core.classifier_v2 import get_classifier

clf = get_classifier()

# Test cases
test_cases = [
    (["E_15", "E_44", "E_45"], "chest/respiratory symptoms"),
    (["E_58", "E_136"], "cardiac symptoms"),
    (["E_169", "E_217"], "GI symptoms"),
]

print("Testing SpecialtyClassifierV2")
print("=" * 50)

for codes, description in test_cases:
    result = clf.predict(codes)
    print(f"\n{description}: {codes}")
    print(f"  -> {result['specialty']} ({result['confidence']:.1%})")
    print(f"  Top 3: {result['top_3']}")

print("\n" + "=" * 50)
info = clf.get_model_info()
print(f"Model Info:")
print(f"  Vocabulary: {info['vocabulary_size']} codes")
print(f"  Accuracy: {info['metrics']['after']['accuracy']:.1%}")
