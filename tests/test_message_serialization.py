"""
unit tests for attack/defense metric dataclass serialization.

verifies that AttackMetrics and DefenseMetrics serialize to dict correctly
(via dataclasses.asdict), since notebooks and pipeline scripts rely on json
serialization of these dataclasses for result persistence.
"""

import json
import os
import sys
from dataclasses import asdict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.benchmarking import AttackMetrics, DefenseMetrics


class TestAttackMetricsSerialization:
    """tests that AttackMetrics serializes correctly to dict and json."""

    def _make_metrics(self, attack_type="agent_poison"):
        m = AttackMetrics(attack_type=attack_type)
        m.total_queries = 20
        m.queries_retrieved_poison = 10
        m.retrievals_with_target_action = 7
        m.successful_task_hijacks = 5
        m.injection_success_rate = 1.0
        m.benign_accuracy = 0.9
        m.execution_time_avg = 0.05
        m.execution_time_std = 0.01
        m.calculate_rates()
        return m

    def test_asdict_contains_all_asr_fields(self):
        """asdict includes asr_r, asr_a, and asr_t keys."""
        m = self._make_metrics()
        d = asdict(m)
        assert "asr_r" in d
        assert "asr_a" in d
        assert "asr_t" in d

    def test_asdict_attack_type_preserved(self):
        """attack_type field is preserved during serialization."""
        m = self._make_metrics("minja")
        d = asdict(m)
        assert d["attack_type"] == "minja"

    def test_asdict_values_are_json_serializable(self):
        """all values produced by asdict can be serialized to json."""
        m = self._make_metrics()
        d = asdict(m)
        serialized = json.dumps(d)
        reloaded = json.loads(serialized)
        assert abs(reloaded["asr_r"] - m.asr_r) < 1e-9

    def test_asr_r_formula_correct(self):
        """asr_r = queries_retrieved_poison / total_queries."""
        m = self._make_metrics()
        expected_asr_r = m.queries_retrieved_poison / m.total_queries
        assert abs(m.asr_r - expected_asr_r) < 1e-9

    def test_asr_a_formula_correct(self):
        """asr_a = retrievals_with_target_action / queries_retrieved_poison."""
        m = self._make_metrics()
        expected_asr_a = m.retrievals_with_target_action / m.queries_retrieved_poison
        assert abs(m.asr_a - expected_asr_a) < 1e-9

    def test_asr_t_formula_correct(self):
        """asr_t = successful_task_hijacks / total_queries (end-to-end rate)."""
        m = self._make_metrics()
        expected_asr_t = m.successful_task_hijacks / m.total_queries
        assert abs(m.asr_t - expected_asr_t) < 1e-9

    def test_all_rates_in_unit_interval(self):
        """all computed rates must be in [0, 1]."""
        m = self._make_metrics()
        assert 0.0 <= m.asr_r <= 1.0
        assert 0.0 <= m.asr_a <= 1.0
        assert 0.0 <= m.asr_t <= 1.0

    @pytest.mark.parametrize("attack_type", ["agent_poison", "minja", "injecmem"])
    def test_roundtrip_json_all_attack_types(self, attack_type):
        """json roundtrip preserves asr values for all attack types."""
        m = self._make_metrics(attack_type)
        d = asdict(m)
        reloaded = json.loads(json.dumps(d))
        assert reloaded["attack_type"] == attack_type
        assert abs(reloaded["asr_r"] - m.asr_r) < 1e-9


class TestDefenseMetricsSerialization:
    """tests that DefenseMetrics serializes correctly to dict and json."""

    def _make_defense_metrics(self, defense_type="watermark"):
        m = DefenseMetrics(defense_type=defense_type)
        m.true_positives = 40
        m.false_positives = 5
        m.true_negatives = 45
        m.false_negatives = 10
        m.total_tests = 40 + 5 + 45 + 10
        m.calculate_rates()
        return m

    def test_asdict_contains_tpr_fpr_f1(self):
        """asdict includes tpr, fpr, and f1_score keys."""
        m = self._make_defense_metrics()
        d = asdict(m)
        assert "tpr" in d
        assert "fpr" in d
        assert "f1_score" in d

    def test_tpr_formula_correct(self):
        """tpr = tp / (tp + fn)."""
        m = self._make_defense_metrics()
        expected_tpr = 40 / (40 + 10)
        assert abs(m.tpr - expected_tpr) < 1e-9

    def test_fpr_formula_correct(self):
        """fpr = fp / (fp + tn)."""
        m = self._make_defense_metrics()
        expected_fpr = 5 / (5 + 45)
        assert abs(m.fpr - expected_fpr) < 1e-9

    def test_f1_formula_correct(self):
        """f1 = 2*tp / (2*tp + fp + fn)."""
        m = self._make_defense_metrics()
        expected_f1 = 2 * 40 / (2 * 40 + 5 + 10)
        assert abs(m.f1_score - expected_f1) < 1e-9

    def test_all_metrics_in_unit_interval(self):
        """tpr, fpr, precision, and f1 must be in [0, 1]."""
        m = self._make_defense_metrics()
        assert 0.0 <= m.tpr <= 1.0
        assert 0.0 <= m.fpr <= 1.0
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.f1_score <= 1.0

    def test_asdict_values_are_json_serializable(self):
        """all values from asdict can be serialized to json without error."""
        m = self._make_defense_metrics()
        d = asdict(m)
        serialized = json.dumps(d)
        reloaded = json.loads(serialized)
        assert abs(reloaded["tpr"] - m.tpr) < 1e-9

    @pytest.mark.parametrize(
        "defense_type", ["watermark", "validation", "proactive", "composite"]
    )
    def test_roundtrip_json_all_defense_types(self, defense_type):
        """json roundtrip preserves metric values for all defense types."""
        m = self._make_defense_metrics(defense_type)
        d = asdict(m)
        reloaded = json.loads(json.dumps(d))
        assert reloaded["defense_type"] == defense_type
        assert abs(reloaded["f1_score"] - m.f1_score) < 1e-9
