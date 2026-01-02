"""Tests for multimodal triage integration."""

import pytest
from app.core.multimodal_fusion import (
    MultimodalFusion,
    TextAnalysis,
    ImageAnalysis,
    RiskLevel,
    RISK_TO_TIER,
)


class TestMultimodalFusion:
    def setup_method(self):
        self.fusion = MultimodalFusion()

    def _make_text(self, urgency="routine", specialty="dermatology"):
        return TextAnalysis(
            specialty=specialty,
            confidence=0.85,
            urgency=urgency,
            differential_diagnosis=[],
            matched_codes=["E_129"],
        )

    def _make_image(self, tier="routine_monitoring", cancer_prob=0.05):
        return ImageAnalysis(
            prediction="NV",
            prediction_label="Melanocytic Nevus",
            confidence=0.90,
            tier=tier,
            tier_display=tier.replace("_", " ").title(),
            cancer_probability=cancer_prob,
            melanoma_probability=0.02,
            reasons=["High confidence benign"],
            message="Appears benign",
        )

    def test_text_only(self):
        text = self._make_text(urgency="routine")
        result = self.fusion.fuse(text, None)
        assert result.agreement_level == "N/A"
        assert "Text-only" in result.reasoning

    def test_image_only(self):
        image = self._make_image(tier="routine_monitoring")
        result = self.fusion.fuse(None, image)
        assert result.agreement_level == "N/A"
        assert result.final_risk_tier == "routine_monitoring"

    def test_both_none(self):
        result = self.fusion.fuse(None, None)
        assert result.final_risk_tier == "consider_evaluation"
        assert "Insufficient" in result.reasoning

    def test_strong_agreement(self):
        text = self._make_text(urgency="routine")
        image = self._make_image(tier="routine_monitoring")
        result = self.fusion.fuse(text, image)
        assert result.agreement_level == "strong"
        assert result.final_risk_tier == "routine_monitoring"

    def test_moderate_agreement(self):
        text = self._make_text(urgency="routine")
        image = self._make_image(tier="consider_evaluation")
        result = self.fusion.fuse(text, image)
        assert result.agreement_level == "moderate"
        assert result.final_risk_tier == "consider_evaluation"

    def test_conflict_image_higher(self):
        text = self._make_text(urgency="routine")
        image = self._make_image(tier="urgent_referral", cancer_prob=0.45)
        result = self.fusion.fuse(text, image)
        assert result.agreement_level == "conflict"
        assert result.final_risk_tier == "urgent_referral"
        assert result.resolution_rationale is not None

    def test_conflict_text_higher(self):
        text = self._make_text(urgency="urgent")
        image = self._make_image(tier="routine_monitoring")
        result = self.fusion.fuse(text, image)
        assert result.agreement_level == "conflict"
        assert result.final_risk_tier == "urgent_referral"

    def test_conservative_fusion(self):
        text = self._make_text(urgency="routine")
        image = self._make_image(tier="routine_referral")
        result = self.fusion.fuse(text, image)
        assert result.final_risk_level >= RiskLevel.ROUTINE_REFERRAL

    def test_cancer_escalation(self):
        text = self._make_text(urgency="routine")
        image = ImageAnalysis(
            prediction="MEL",
            prediction_label="Melanoma",
            confidence=0.75,
            tier="urgent_referral",
            tier_display="Urgent Referral",
            cancer_probability=0.75,
            melanoma_probability=0.75,
            reasons=["Melanoma detected"],
            message="Urgent evaluation needed",
        )
        result = self.fusion.fuse(text, image)
        assert result.final_risk_tier == "urgent_referral"
        assert result.final_urgency == "urgent"


class TestRiskMappings:
    def test_all_tiers_mapped(self):
        tiers = ["routine_monitoring", "consider_evaluation", "routine_referral", "urgent_referral"]
        for tier in tiers:
            assert tier in RISK_TO_TIER.values()

    def test_risk_ordering(self):
        assert RiskLevel.ROUTINE_MONITORING < RiskLevel.CONSIDER_EVALUATION
        assert RiskLevel.CONSIDER_EVALUATION < RiskLevel.ROUTINE_REFERRAL
        assert RiskLevel.ROUTINE_REFERRAL < RiskLevel.URGENT_REFERRAL
        assert RiskLevel.URGENT_REFERRAL < RiskLevel.EMERGENCY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
