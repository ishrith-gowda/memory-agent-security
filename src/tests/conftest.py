"""
test configuration and fixtures for memory agent security research.

this module provides:
- pytest configuration
- test fixtures for common test data
- mock objects for external dependencies
- test utilities and helpers

all comments are lowercase.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attacks.implementations import AttackSuite, create_attack
from defenses.implementations import DefenseSuite, create_defense
from evaluation.benchmarking import (AttackMetrics, BenchmarkRunner,
                                     DefenseMetrics)
from memory_systems.wrappers import create_memory_system
from utils.config import configmanager
from utils.logging import logger as research_logger
from utils.logging import setup_experiment_logging
from watermark.watermarking import ProvenanceTracker, create_watermark_encoder


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def test_config(temp_dir: Path) -> configmanager:
    """create a test configuration manager."""
    config_dir = temp_dir / "configs"
    config_dir.mkdir()

    # Create test config files
    memory_config = {
        "mem0": {"api_key": "test_key", "collection": "test_collection"},
        "amem": {"config_path": str(config_dir / "amem_config.yaml")},
        "memgpt": {"agent_id": "test_agent", "server_url": "http://localhost:8080"},
    }

    experiment_config = {
        "attacks": {
            "agent_poison": {"intensity": 0.5},
            "minja": {"injection_rate": 0.3},
            "injecmem": {"manipulation_level": 2},
        },
        "defenses": {
            "watermark": {"encoder_type": "lsb"},
            "validation": {"strict_mode": False},
            "proactive": {"simulation_depth": 3},
        },
        "evaluation": {
            "num_trials": 5,
            "confidence_threshold": 0.8,
            "performance_metrics": ["asr", "tpr", "fpr"],
        },
    }

    # Write config files
    import yaml

    with open(config_dir / "memory.yaml", "w") as f:
        yaml.dump(memory_config, f)

    with open(config_dir / "experiment.yaml", "w") as f:
        yaml.dump(experiment_config, f)

    # Create config manager
    config = configmanager(str(config_dir))
    return config


@pytest.fixture(scope="session")
def test_logger(temp_dir: Path) -> research_logger:
    """create a test logger."""
    log_dir = temp_dir / "logs"
    log_dir.mkdir()

    logger_instance = setup_experiment_logging("test_experiment")
    return logger_instance


@pytest.fixture
def mock_memory_system() -> Mock:
    """create a mock memory system for testing."""
    mock_memory = Mock()
    mock_memory.store.return_value = {"status": "success", "id": "test_id"}
    mock_memory.retrieve.return_value = {"content": "test content", "metadata": {}}
    mock_memory.search.return_value = [
        {"id": "test_id", "content": "test content", "score": 0.9}
    ]
    mock_memory.get_all_keys.return_value = ["key1", "key2", "key3"]
    return mock_memory


@pytest.fixture
def mock_mem0_wrapper(mock_memory_system: Mock) -> Mock:
    """create a mock Mem0 wrapper."""
    with patch("src.memory_systems.wrappers.Mem0Memory") as mock_mem0:
        mock_mem0.return_value = mock_memory_system
        wrapper = create_memory_system("mem0", {"user_id": "test"})
        yield wrapper


@pytest.fixture
def mock_amem_wrapper(mock_memory_system: Mock) -> Mock:
    """create a mock A-MEM wrapper."""
    with patch("src.memory_systems.wrappers.AgenticMemory") as mock_amem:
        mock_amem.return_value = mock_memory_system
        wrapper = create_memory_system("amem", {"config": "test"})
        yield wrapper


@pytest.fixture
def mock_memgpt_wrapper(mock_memory_system: Mock) -> Mock:
    """create a mock MemGPT wrapper."""
    with patch("src.memory_systems.wrappers.Letta") as mock_memgpt:
        mock_memgpt.return_value = mock_memory_system
        wrapper = create_memory_system("memgpt", {"agent_id": "test"})
        yield wrapper


@pytest.fixture
def test_attack_suite() -> AttackSuite:
    """create a test attack suite."""
    return AttackSuite()


@pytest.fixture
def test_defense_suite() -> DefenseSuite:
    """create a test defense suite."""
    return DefenseSuite()


@pytest.fixture
def test_watermark_encoders() -> Dict[str, Any]:
    """create test watermark encoders."""
    encoders = {}
    for encoder_type in ["lsb", "semantic", "crypto", "composite"]:
        encoders[encoder_type] = create_watermark_encoder(encoder_type)
    return encoders


@pytest.fixture
def test_provenance_tracker() -> ProvenanceTracker:
    """create a test provenance tracker."""
    return ProvenanceTracker()


@pytest.fixture
def test_benchmark_runner(
    test_config: configmanager, test_logger: ResearchLogger
) -> BenchmarkRunner:
    """create a test benchmark runner."""
    return BenchmarkRunner(config=test_config, logger=test_logger)


@pytest.fixture
def sample_memory_content() -> List[Any]:
    """provide sample memory content for testing."""
    return [
        "This is a simple text memory entry.",
        "Another memory entry with different content.",
        {
            "type": "structured",
            "content": "Structured memory data",
            "metadata": {"source": "test"},
        },
        ["list", "of", "memory", "items", "for", "testing"],
        "Memory entry with special characters: !@#$%^&*()",
        "A very long memory entry that contains a lot of text to test how the system handles larger content sizes and whether it can properly process and store this information without any issues or truncation problems.",
        {
            "type": "conversation",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        },
        "Edge case: empty content",
        "Edge case: content with unicode: 你好世界 🌍",
        "Edge case: content with newlines\nand\ntabs\tand\tother\twhitespace",
    ]


@pytest.fixture
def sample_attack_payloads() -> Dict[str, Any]:
    """provide sample attack payloads for testing."""
    return {
        "agent_poison": {
            "corruption_patterns": ["override", "inject", "malicious"],
            "intensity": 0.7,
        },
        "minja": {
            "injection_content": "MALICIOUS_PAYLOAD",
            "target_fields": ["content", "metadata"],
            "injection_rate": 0.5,
        },
        "injecmem": {
            "manipulation_type": "insertion",
            "target_positions": [0, -1],  # Beginning and end
            "manipulation_level": 3,
        },
    }


@pytest.fixture
def sample_watermarks() -> List[str]:
    """provide sample watermarks for testing."""
    return [
        "test_watermark_001",
        "security_research_2024",
        "provenance_tracking_test",
        "composite_watermark_validation",
        "cryptographic_signature_test",
        "semantic_embedding_watermark",
        "lsb_steganography_test",
        "multi_layer_protection_test",
    ]


@pytest.fixture
def performance_test_data() -> Dict[str, Any]:
    """provide data for performance testing."""
    return {
        "small_content": ["Short test"] * 10,
        "medium_content": ["Medium sized test content for benchmarking"] * 50,
        "large_content": [
            "Very large test content that simulates real-world memory entries with substantial amounts of text data for comprehensive performance evaluation"
        ]
        * 100,
        "mixed_content": [
            "text",
            {"structured": "data"},
            ["list", "content"],
            "unicode: 测试内容",
            "special: !@#$%^&*()",
        ]
        * 20,
    }


# Test utilities
def assert_attack_result_structure(result: Dict[str, Any], attack_type: str):
    """assert that attack result has correct structure."""
    assert isinstance(result, dict)
    assert "attack_type" in result
    assert result["attack_type"] == attack_type
    assert "success" in result
    assert "timestamp" in result
    assert isinstance(result["success"], bool)
    assert isinstance(result["timestamp"], (int, float))


def assert_defense_result_structure(result: Dict[str, Any], defense_type: str):
    """assert that defense result has correct structure."""
    assert isinstance(result, dict)
    assert "defense_type" in result
    assert result["defense_type"] == defense_type
    assert "attack_detected" in result
    assert "confidence" in result
    assert "timestamp" in result
    assert isinstance(result["attack_detected"], bool)
    assert isinstance(result["confidence"], (int, float))
    assert isinstance(result["timestamp"], (int, float))


def assert_watermark_operation(content: str, watermarked: str, watermark: str):
    """assert that watermark operation was successful."""
    assert isinstance(watermarked, str)
    assert len(watermarked) > 0
    # Content should be preserved (may be modified but not destroyed)
    assert len(watermarked) >= len(content) * 0.8  # Allow some overhead


def assert_metrics_structure(metrics: Any, metric_type: str):
    """assert that metrics have correct structure."""
    assert metrics is not None

    if metric_type == "attack":
        assert hasattr(metrics, "attack_type")
        assert hasattr(metrics, "total_attempts")
        assert hasattr(metrics, "successful_attempts")
        assert hasattr(metrics, "asr_r")
        assert hasattr(metrics, "asr_a")
        assert hasattr(metrics, "asr_t")
        assert hasattr(metrics, "execution_time_avg")
    elif metric_type == "defense":
        assert hasattr(metrics, "defense_type")
        assert hasattr(metrics, "total_tests")
        assert hasattr(metrics, "true_positives")
        assert hasattr(metrics, "false_positives")
        assert hasattr(metrics, "tpr")
        assert hasattr(metrics, "fpr")
        assert hasattr(metrics, "precision")
        assert hasattr(metrics, "recall")


def create_mock_memory_operation(
    content: Any, operation: str = "store"
) -> Dict[str, Any]:
    """create a mock memory operation result."""
    base_result = {"operation": operation, "timestamp": time.time(), "success": True}

    if operation == "store":
        base_result.update(
            {
                "id": f"mock_id_{hash(str(content)) % 1000}",
                "content_hash": hash(str(content)),
                "size": len(str(content)),
            }
        )
    elif operation == "retrieve":
        base_result.update(
            {
                "content": content,
                "metadata": {"source": "mock", "timestamp": base_result["timestamp"]},
            }
        )
    elif operation == "search":
        base_result.update(
            {
                "results": [{"id": "mock_id", "content": content, "score": 0.95}],
                "total_matches": 1,
                "search_time": 0.001,
            }
        )

    return base_result


# Pytest configuration
def pytest_configure(config):
    """configure pytest for memory security tests."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests for basic functionality"
    )


def pytest_collection_modifyitems(config, items):
    """modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance)

        # Mark integration tests
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)

        # Mark smoke tests
        if "smoke" in item.name or "basic" in item.name:
            item.add_marker(pytest.mark.smoke)


# Custom pytest fixtures for specific test scenarios
@pytest.fixture
def clean_memory_state():
    """ensure clean memory state for tests."""
    # This would reset any global state if needed
    yield
    # Cleanup after test


@pytest.fixture
def isolated_test_environment(temp_dir: Path, monkeypatch):
    """create an isolated test environment."""
    # Set environment variables for testing
    monkeypatch.setenv("MEMORY_SECURITY_TEST", "true")
    monkeypatch.setenv("DISABLE_EXTERNAL_APIS", "true")

    # Change to temp directory
    monkeypatch.chdir(temp_dir)

    yield temp_dir


if __name__ == "__main__":
    print("test configuration and fixtures loaded successfully")
