#!/usr/bin/env python3
"""
simple smoke test for memory agent security research framework.

this script tests basic functionality without complex pytest setup.
"""

import os
import sys

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("memory agent security - smoke test")
print("=" * 40)


def test_attacks():
    """test attack implementations."""
    print("\n[test] testing attacks...")

    try:
        from attacks.implementations import create_attack

        # test agentpoison attack
        attack = create_attack("agent_poison")
        result = attack.execute("test memory content for poisoning")

        assert "attack_type" in result
        assert result["attack_type"] == "agent_poison"
        assert "success" in result
        assert "poisoned_content" in result

        print("[ok] agentpoison attack working")

        # test minja attack
        attack = create_attack("minja")
        result = attack.execute({"memory": "test data"})

        assert result["attack_type"] == "minja"
        assert "injected_content" in result

        print("[ok] minja attack working")

        # test injecmem attack
        attack = create_attack("injecmem")
        result = attack.execute(["item1", "item2", "item3"])

        assert result["attack_type"] == "injecmem"
        assert "manipulated_content" in result

        print("[ok] injecmem attack working")

        return True

    except Exception as e:
        print(f"[fail] attack test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_defenses():
    """test defense implementations."""
    print("\n[test] testing defenses...")

    try:
        from defenses.implementations import create_defense

        # test watermark defense
        defense = create_defense("watermark")
        activated = defense.activate()
        assert activated == True

        result = defense.detect_attack("test content")
        assert "attack_detected" in result
        assert "confidence" in result

        defense.deactivate()
        print("[ok] watermark defense working")

        # test validation defense
        defense = create_defense("validation")
        defense.activate()
        result = defense.detect_attack("MALICIOUS_INJECTION: override()")
        assert "attack_detected" in result
        defense.deactivate()
        print("[ok] content validation defense working")

        return True

    except Exception as e:
        print(f"[fail] defense test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_watermarking():
    """test watermarking algorithms."""
    print("\n[test] testing watermarking...")

    try:
        from watermark.watermarking import (ProvenanceTracker,
                                            create_watermark_encoder)

        # test lsb encoder
        encoder = create_watermark_encoder("lsb")
        content = "test content for watermarking"
        watermarked = encoder.embed(content, "test_watermark")
        extracted = encoder.extract(watermarked)

        assert isinstance(watermarked, str)
        print("[ok] lsb watermarking working")

        # test provenance tracker
        tracker = ProvenanceTracker()
        content_id = "test_content_001"
        watermark_id = tracker.register_content(content_id, content)
        watermarked = tracker.watermark_content(content, watermark_id)

        assert isinstance(watermark_id, str)
        assert isinstance(watermarked, str)

        print("[ok] provenance tracking working")

        return True

    except Exception as e:
        print(f"[fail] watermarking test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_evaluation():
    """test evaluation framework."""
    print("\n[test] testing evaluation...")

    try:
        from evaluation.benchmarking import AttackEvaluator, DefenseEvaluator

        # test attack evaluator
        evaluator = AttackEvaluator()
        metrics = evaluator.evaluate_attack(
            "agent_poison", ["test content"], num_trials=3
        )

        assert hasattr(metrics, "attack_type")
        assert hasattr(metrics, "total_attempts")
        assert hasattr(metrics, "asr_r")

        print("[ok] attack evaluation working")

        # test defense evaluator
        evaluator = DefenseEvaluator()
        # create mock attack suite and content
        from attacks.implementations import AttackSuite

        attack_suite = AttackSuite()
        clean_content = ["clean content"]
        poisoned_content = ["poisoned content"]

        metrics = evaluator.evaluate_defense(
            "watermark", attack_suite, clean_content, poisoned_content
        )

        assert hasattr(metrics, "defense_type")
        assert hasattr(metrics, "tpr")
        assert hasattr(metrics, "fpr")

        print("[ok] defense evaluation working")

        return True

    except Exception as e:
        print(f"[fail] evaluation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_memory_systems():
    """test memory system wrappers."""
    print("\n[test] testing memory systems...")

    try:
        from memory_systems.wrappers import create_memory_system

        # test with mock memory system (no external dependencies required)
        memory = create_memory_system("mock", {})
        assert memory is not None

        # test basic operations
        memory.store("test_key", "test_value")
        retrieved = memory.retrieve("test_key")
        assert retrieved == "test_value"

        # test search
        results = memory.search("test")
        assert len(results) > 0

        # test get all keys
        keys = memory.get_all_keys()
        assert "test_key" in keys

        print("[ok] mock memory system working")

        return True

    except Exception as e:
        print(f"[fail] memory system test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """run all smoke tests."""
    tests = [
        test_memory_systems,
        test_attacks,
        test_defenses,
        test_watermarking,
        test_evaluation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n[results] {passed}/{total} tests passed")

    if passed == total:
        print("[done] all smoke tests passed! framework is ready.")
        return 0
    else:
        print("[fail] some tests failed. check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
