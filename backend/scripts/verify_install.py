"""
Verify the installation is working correctly.
Run this after setup to ensure everything works.
"""
import sys
from pathlib import Path

def check_imports():
    """Check all required packages can be imported."""
    print("Checking imports...")
    errors = []
    
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("structlog", "structlog"),
    ]
    
    for import_name, package_name in packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError as e:
            print(f"  ❌ {package_name}: {e}")
            errors.append(package_name)
    
    return len(errors) == 0


def check_data_files():
    """Check required data files exist."""
    print("\nChecking data files...")
    
    base = Path(__file__).parent.parent.parent
    
    required_files = [
        ("data/ddxplus/release_evidences.json", "DDXPlus evidences"),
        ("data/ddxplus/release_conditions.json", "DDXPlus conditions"),
        ("data/ddxplus/symptom_condition_probs.json", "Symptom probabilities"),
        ("backend/data/classifier/model.pkl", "XGBoost model"),
        ("backend/data/classifier/vocabulary.pkl", "Vocabulary"),
    ]
    
    missing = []
    for filepath, desc in required_files:
        full_path = base / filepath
        if full_path.exists():
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ {desc} ({filepath})")
            missing.append(filepath)
    
    return len(missing) == 0


def check_pipeline():
    """Test the triage pipeline."""
    print("\nTesting pipeline...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from app.core.triage_pipeline_v2 import TriagePipelineV2
        
        base = Path(__file__).parent.parent.parent
        
        pipeline = TriagePipelineV2()
        pipeline.load(
            evidences_path=base / "data/ddxplus/release_evidences.json",
            model_path=base / "backend/data/classifier/model.pkl",
            vocab_path=base / "backend/data/classifier/vocabulary.pkl",
            enable_explanations=False,
        )
        
        # Test prediction
        result = pipeline.predict(
            ["chest pain", "shortness of breath"],
            include_ddx=False,
            include_explanation=False,
        )
        
        pipeline.unload()
        
        print(f"  ✅ Pipeline loaded successfully")
        print(f"  ✅ Test prediction: {result['specialty']} ({result['confidence']:.0%})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline error: {e}")
        return False


def main():
    print("=" * 50)
    print("Medical Triage AI - Installation Verification")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", check_imports()))
    results.append(("Data files", check_data_files()))
    results.append(("Pipeline", check_pipeline()))
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n✅ All checks passed! System is ready.")
        print("\nTo start the API server:")
        print("  cd backend")
        print("  uvicorn app.main:app --reload")
        print("\nOr use: make run")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
