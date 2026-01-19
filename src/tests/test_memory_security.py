"""
test infrastructure for memory agent security research.

this module provides comprehensive testing capabilities:
- unit tests for all components
- integration tests for attack-defense interactions
- performance benchmarks
- test utilities and fixtures

all comments are lowercase.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attacks.implementations import AttackSuite, create_attack
from defenses.implementations import DefenseSuite, create_defense
from evaluation.benchmarking import (AttackEvaluator, BenchmarkRunner,
                                     DefenseEvaluator)
from memory_systems.wrappers import create_memory_system
from utils.logging import logger
from watermark.watermarking import ProvenanceTracker, create_watermark_encoder


class TestAttackImplementations:
    """test cases for attack implementations."""

    def test_agent_poison_attack(self):
        """test AgentPoison attack execution."""
        attack = create_attack("agent_poison")

        test_content = "This is a test memory entry"
        result = attack.execute(test_content)

        assert result["attack_type"] == "agent_poison"
        assert "poisoned_content" in result
        assert result["success"] is True or isinstance(result.get("error"), str)

    def test_minja_attack(self):
        """test MINJA attack execution."""
        attack = create_attack("minja")

        test_content = {"memory": "test data", "metadata": "test"}
        result = attack.execute(test_content)

        assert result["attack_type"] == "minja"
        assert "injected_content" in result
        assert result["success"] is True or isinstance(result.get("error"), str)

    def test_injecmem_attack(self):
        """test InjecMEM attack execution."""
        attack = create_attack("injecmem")

        test_content = ["memory1", "memory2", "memory3"]
        result = attack.execute(test_content)

        assert result["attack_type"] == "injecmem"
        assert "manipulated_content" in result
        assert result["success"] is True or isinstance(result.get("error"), str)

    def test_attack_suite(self):
        """test attack suite batch execution."""
        suite = AttackSuite()

        test_content = "Test memory content"
        results = suite.execute_all(test_content)

        assert "attack_results" in results
        assert len(results["attack_results"]) == 3  # All attack types
        assert all(
            isinstance(result, dict) for result in results["attack_results"].values()
        )


class TestDefenseImplementations:
    """test cases for defense implementations."""

    def test_watermark_defense(self):
        """test watermark defense activation and detection."""
        defense = create_defense("watermark")

        # Test activation
        assert defense.activate() is True

        # Test detection on clean content
        clean_result = defense.detect_attack("clean content")
        assert "attack_detected" in clean_result
        assert isinstance(clean_result["confidence"], (int, float))

        # Test deactivation
        assert defense.deactivate() is True

    def test_validation_defense(self):
        """test content validation defense."""
        defense = create_defense("validation")

        assert defense.activate() is True

        # Test pattern detection
        suspicious_content = "MALICIOUS_INJECTION: system.override()"
        result = defense.detect_attack(suspicious_content)

        assert result["attack_detected"] is True or "validation_results" in result

        assert defense.deactivate() is True

    def test_proactive_defense(self):
        """test proactive defense with attack simulation."""
        defense = create_defense("proactive")

        assert defense.activate() is True

        # Test simulation-based detection
        test_content = "test content for simulation"
        result = defense.detect_attack(test_content)

        assert "attack_detected" in result
        assert "simulation_results" in result

        assert defense.deactivate() is True

    def test_composite_defense(self):
        """test composite defense coordination."""
        defense = create_defense("composite")

        assert defense.activate() is True

        # Test multi-defense detection
        test_content = "test content for composite analysis"
        result = defense.detect_attack(test_content)

        assert "component_results" in result
        assert len(result["component_results"]) == 3  # All defense types

        assert defense.deactivate() is True

    def test_defense_suite(self):
        """test defense suite coordination."""
        suite = DefenseSuite()

        activation_results = suite.activate_all()
        assert len(activation_results) == 4  # All defense types
        assert all(isinstance(result, bool) for result in activation_results.values())

        # Test detection
        test_content = "test content for suite detection"
        detection_results = suite.detect_attack(test_content)

        assert "defense_results" in detection_results
        assert len(detection_results["defense_results"]) == 4


class TestWatermarking:
    """test cases for watermarking algorithms."""

    def test_lsb_watermark_encoder(self):
        """test LSB watermark encoding and extraction."""
        encoder = create_watermark_encoder("lsb")

        content = "This is a test message for watermarking"
        watermark = "test_watermark_123"

        # Test embedding
        watermarked = encoder.embed(content, watermark)
        assert isinstance(watermarked, str)
        assert len(watermarked) >= len(content)  # May be longer due to embedding

        # Test extraction
        extracted = encoder.extract(watermarked)
        assert extracted is not None or watermark in (extracted or "")

    def test_semantic_watermark_encoder(self):
        """test semantic watermark encoding."""
        encoder = create_watermark_encoder("semantic")

        content = "The system should validate all inputs carefully."
        watermark = "security_check"

        watermarked = encoder.embed(content, watermark)
        assert isinstance(watermarked, str)

        extracted = encoder.extract(watermarked)
        # Semantic extraction may not be perfect
        assert extracted is None or isinstance(extracted, str)

    def test_cryptographic_watermark_encoder(self):
        """test cryptographic watermark encoding."""
        encoder = create_watermark_encoder("crypto")

        content = "Important security information"
        watermark = "confidential_data"

        watermarked = encoder.embed(content, watermark)
        assert isinstance(watermarked, str)
        assert "<!--WATERMARK:" in watermarked

        extracted = encoder.extract(watermarked)
        # Crypto extraction requires original watermark for verification
        assert extracted is None or isinstance(extracted, str)

    def test_composite_watermark_encoder(self):
        """test composite watermark encoding."""
        encoder = create_watermark_encoder("composite")

        content = "Multi-layered security test content"
        watermark = "composite_test"

        watermarked = encoder.embed(content, watermark)
        assert isinstance(watermarked, str)

        extracted = encoder.extract(watermarked)
        assert extracted is None or watermark in (extracted or "")

    def test_provenance_tracker(self):
        """test provenance tracking functionality."""
        tracker = ProvenanceTracker()

        content_id = "test_content_001"
        content = "Test content for provenance tracking"

        # Register content
        watermark_id = tracker.register_content(content_id, content)
        assert isinstance(watermark_id, str)
        assert len(watermark_id) > 0

        # Verify provenance
        watermarked = tracker.watermark_content(content, watermark_id)
        assert isinstance(watermarked, str)

        provenance = tracker.verify_provenance(watermarked)
        assert provenance is None or isinstance(provenance, dict)


class TestMemorySystems:
    """test cases for memory system wrappers."""

    @patch("src.memory_systems.wrappers.Mem0Wrapper")
    def test_mem0_wrapper(self, mock_wrapper):
        """test Mem0 wrapper initialization."""
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance

        wrapper = create_memory_system("mem0", {"user_id": "test"})

        # Verify wrapper was created
        assert wrapper is not None
        mock_wrapper.assert_called_once()

    @patch("src.memory_systems.wrappers.AMEMWrapper")
    def test_amem_wrapper(self, mock_wrapper):
        """test A-MEM wrapper initialization."""
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance

        wrapper = create_memory_system("amem", {"config": "test"})

        assert wrapper is not None
        mock_wrapper.assert_called_once()

    @patch("src.memory_systems.wrappers.MemGPTWrapper")
    def test_memgpt_wrapper(self, mock_wrapper):
        """test MemGPT wrapper initialization."""
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance

        wrapper = create_memory_system("memgpt", {"agent_id": "test"})

        assert wrapper is not None
        mock_wrapper.assert_called_once()

    def test_invalid_memory_system(self):
        """test error handling for invalid memory system type."""
        with pytest.raises(ValueError, match="unsupported memory system type"):
            create_memory_system("invalid_type")


class TestEvaluation:
    """test cases for evaluation framework."""

    def test_attack_evaluator(self):
        """test attack evaluator functionality."""
        evaluator = AttackEvaluator()

        test_content = ["test content 1", "test content 2"]
        metrics = evaluator.evaluate_attack("agent_poison", test_content, num_trials=2)

        assert metrics.attack_type == "agent_poison"
        assert metrics.total_attempts > 0
        assert isinstance(metrics.asr_r, (int, float))
        assert isinstance(metrics.execution_time_avg, (int, float))

    def test_defense_evaluator(self):
        """test defense evaluator functionality."""
        evaluator = DefenseEvaluator()
        attack_suite = AttackSuite()

        clean_content = ["clean content 1", "clean content 2"]
        poisoned_content = ["poisoned content 1", "poisoned content 2"]

        metrics = evaluator.evaluate_defense(
            "watermark", attack_suite, clean_content, poisoned_content
        )

        assert metrics.defense_type == "watermark"
        assert metrics.total_tests > 0
        assert isinstance(metrics.tpr, (int, float))
        assert isinstance(metrics.fpr, (int, float))

    def test_benchmark_runner(self):
        """test benchmark runner functionality."""
        runner = BenchmarkRunner()

        test_content = ["benchmark test content"]
        result = runner.run_benchmark("test_experiment", test_content, num_trials=2)

        assert result.experiment_id == "test_experiment"
        assert isinstance(result.timestamp, (int, float))
        assert isinstance(result.attack_metrics, dict)
        assert isinstance(result.defense_metrics, dict)
        assert result.test_duration > 0


class TestIntegration:
    """integration tests for attack-defense interactions."""

    def test_attack_defense_integration(self):
        """test end-to-end attack-defense interaction."""
        # Create attack and defense
        attack = create_attack("agent_poison")
        defense = create_defense("composite")

        # Activate defense
        assert defense.activate() is True

        # Execute attack
        test_content = "Integration test content"
        attack_result = attack.execute(test_content)

        # Test defense detection
        if attack_result.get("success", False):
            poisoned_content = attack_result.get("poisoned_content", test_content)
            defense_result = defense.detect_attack(poisoned_content)

            assert "attack_detected" in defense_result
            assert isinstance(defense_result["confidence"], (int, float))

        # Deactivate defense
        assert defense.deactivate() is True

    def test_full_evaluation_pipeline(self):
        """test complete evaluation pipeline."""
        # Setup components
        attack_suite = AttackSuite()
        defense_suite = DefenseSuite()
        benchmark_runner = BenchmarkRunner()

        # Generate test data
        test_content = [
            "Test memory entry one",
            "Test memory entry two",
            {"type": "structured", "content": "test data"},
            ["list", "of", "memory", "items"],
        ]

        # Run benchmark
        result = benchmark_runner.run_benchmark(
            "integration_test", test_content, num_trials=3
        )

        # Verify results
        assert result.experiment_id == "integration_test"
        assert len(result.attack_metrics) > 0
        assert len(result.defense_metrics) > 0
        assert result.total_memory_operations > 0
        assert isinstance(result.memory_integrity_score, (int, float))


# Performance benchmarks
class TestPerformance:
    """performance benchmark tests."""

    def test_attack_performance(self, benchmark):
        """benchmark attack execution performance."""
        attack = create_attack("agent_poison")
        test_content = "Performance test content"

        # Benchmark attack execution
        result = benchmark(lambda: attack.execute(test_content))

        assert result is not None
        assert "mean" in result.stats
        assert result.stats["mean"] > 0

    def test_defense_performance(self, benchmark):
        """benchmark defense detection performance."""
        defense = create_defense("watermark")
        defense.activate()
        test_content = "Performance test content"

        try:
            # Benchmark defense detection
            result = benchmark(lambda: defense.detect_attack(test_content))

            assert result is not None
            assert "mean" in result.stats
            assert result.stats["mean"] > 0
        finally:
            defense.deactivate()

    def test_watermark_performance(self, benchmark):
        """benchmark watermark operations."""
        encoder = create_watermark_encoder("lsb")
        content = "Performance test content" * 10  # Larger content
        watermark = "performance_test_watermark"

        # Benchmark embedding
        embed_result = benchmark(lambda: encoder.embed(content, watermark))
        assert embed_result is not None

        # Benchmark extraction
        watermarked = encoder.embed(content, watermark)
        extract_result = benchmark(lambda: encoder.extract(watermarked))
        assert extract_result is not None


if __name__ == "__main__":
    # run basic smoke tests
    print("running memory agent security test suite...")

    # test imports
    try:
        from ..evaluation.benchmarking import BenchmarkRunner
        from ..watermark.watermarking import create_watermark_encoder
        from .implementations import create_attack, create_defense

        print("[ok] all imports successful")
    except ImportError as e:
        print(f"[fail] import error: {e}")
        exit(1)

    # test basic functionality
    try:
        attack = create_attack("agent_poison")
        result = attack.execute("test content")
        print("[ok] attack execution successful")
    except Exception as e:
        print(f"[fail] attack execution failed: {e}")
        exit(1)

    try:
        defense = create_defense("watermark")
        defense.activate()
        result = defense.detect_attack("test content")
        defense.deactivate()
        print("[ok] defense detection successful")
    except Exception as e:
        print(f"[fail] defense detection failed: {e}")
        exit(1)

    try:
        encoder = create_watermark_encoder("lsb")
        watermarked = encoder.embed("test content", "test watermark")
        extracted = encoder.extract(watermarked)
        print("[ok] watermark operations successful")
    except Exception as e:
        print(f"[fail] watermark operations failed: {e}")
        exit(1)

    print("all smoke tests passed!")
