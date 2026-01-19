"""
defense implementations for memory agent security.

this module implements defense mechanisms against memory attacks:
- provenance-aware watermarking defenses
- attack detection and mitigation
- content validation and recovery

all comments are lowercase.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from attacks.implementations import AttackSuite, create_attack
from defenses.base import Defense
from memory_systems.wrappers import create_memory_system
from utils.logging import logger
from watermark.watermarking import ProvenanceTracker, create_watermark_encoder


class WatermarkDefense(Defense):
    """
    watermark-based defense against memory attacks.

    uses watermarking techniques to detect and prevent unauthorized
    memory modifications and injection attacks.
    """

    @property
    def defense_type(self) -> str:
        """the type of defense."""
        return "watermark"

    @property
    def protected_attacks(self) -> List[str]:
        """list of attack types this defense can protect against."""
        return ["agent_poison", "minja", "injecmem"]

    @property
    def description(self) -> str:
        """human-readable description of what the defense does."""
        return "watermark-based provenance tracking and attack detection"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize watermark defense.

        args:
            config: defense configuration
        """
        super().__init__("watermark", config)
        self.encoder_type = self.config.get("encoder_type", "composite")
        self.detection_threshold = self.config.get("detection_threshold", 0.7)

        # Initialize watermark encoder
        self.encoder = create_watermark_encoder(
            self.encoder_type, self.config.get("encoder_config", {})
        )

        # Initialize provenance tracker
        self.tracker = ProvenanceTracker(self.config.get("tracker_config", {}))

        self.logger = logger

    def activate(self, **kwargs) -> bool:
        """
        activate watermark defense.

        args:
            **kwargs: activation parameters

        returns:
            True if activation successful
        """
        try:
            self.logger.logger.info("activating watermark defense")
            # Defense is ready to use
            return True
        except Exception as e:
            self.logger.log_error("watermark_activate", e)
            return False

    def deactivate(self, **kwargs) -> bool:
        """
        deactivate watermark defense.

        args:
            **kwargs: deactivation parameters

        returns:
            True if deactivation successful
        """
        try:
            self.logger.logger.info("deactivating watermark defense")
            return True
        except Exception as e:
            self.logger.log_error("watermark_deactivate", e)
            return False

    def detect_attack(
        self, content: Any, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        detect attacks using watermark analysis.

        args:
            content: content to analyze
            context: additional context

        returns:
            detection result
        """
        start_time = time.time()

        try:
            # Check for watermark presence
            provenance = self.tracker.verify_provenance(str(content))

            if not provenance:
                # No watermark found - potential attack
                detection_result = {
                    "attack_detected": True,
                    "detection_method": "missing_watermark",
                    "confidence": 0.9,
                    "reason": "content lacks expected provenance watermark",
                }
            else:
                # Check watermark integrity
                confidence = provenance.get("confidence", 0.0)
                if confidence < self.detection_threshold:
                    detection_result = {
                        "attack_detected": True,
                        "detection_method": "watermark_tampering",
                        "confidence": 1.0 - confidence,
                        "reason": f"watermark confidence too low: {confidence:.2f}",
                    }
                else:
                    detection_result = {
                        "attack_detected": False,
                        "detection_method": "watermark_verification",
                        "confidence": confidence,
                        "provenance": provenance,
                    }

            execution_time = time.time() - start_time
            detection_result["execution_time"] = execution_time

            self.logger.log_defense_activation(
                self.defense_type,
                {"detection_method": detection_result["detection_method"]},
            )

            return detection_result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.log_error(
                "watermark_detect", e, {"content": str(content)[:100]}
            )

            return {
                "attack_detected": False,  # Default to no attack on error
                "error": str(e),
                "execution_time": execution_time,
            }

    def validate_compatibility(self, attack_type: str) -> bool:
        """
        validate compatibility with attack type.

        args:
            attack_type: type of attack to defend against

        returns:
            True if compatible
        """
        # Watermark defense is compatible with all attack types
        return True


class ContentValidationDefense(Defense):
    """
    content validation defense against memory attacks.

    validates memory content integrity using multiple validation
    techniques including checksums, pattern analysis, and anomaly detection.
    """

    @property
    def defense_type(self) -> str:
        """the type of defense."""
        return "content_validation"

    @property
    def protected_attacks(self) -> List[str]:
        """list of attack types this defense can protect against."""
        return ["agent_poison", "minja", "injecmem"]

    @property
    def description(self) -> str:
        """human-readable description of what the defense does."""
        return "validates memory content integrity using checksums and pattern analysis"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize content validation defense.

        args:
            config: defense configuration
        """
        super().__init__("content_validation", config)
        self.validation_methods = self.config.get(
            "validation_methods",
            ["checksum_verification", "pattern_analysis", "anomaly_detection"],
        )
        self.checksum_algorithm = self.config.get("checksum_algorithm", "sha256")

        # Known good patterns and checksums
        self.known_checksums: Dict[str, str] = {}
        self.known_patterns: List[str] = self.config.get("known_patterns", [])

        self.logger = logger

    def activate(self, **kwargs) -> bool:
        """
        activate content validation defense.

        args:
            **kwargs: activation parameters

        returns:
            True if activation successful
        """
        try:
            self.logger.logger.info("activating content validation defense")
            # Load known good checksums and patterns if provided
            return True
        except Exception as e:
            self.logger.log_error("validation_activate", e)
            return False

    def deactivate(self, **kwargs) -> bool:
        """
        deactivate content validation defense.

        args:
            **kwargs: deactivation parameters

        returns:
            True if deactivation successful
        """
        try:
            self.logger.logger.info("deactivating content validation defense")
            return True
        except Exception as e:
            self.logger.log_error("validation_deactivate", e)
            return False

    def detect_attack(
        self, content: Any, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        detect attacks using content validation.

        args:
            content: content to validate
            context: validation context

        returns:
            validation result
        """
        start_time = time.time()

        try:
            content_str = str(content)
            validation_results = {}

            # Checksum verification
            if "checksum_verification" in self.validation_methods:
                current_checksum = self._calculate_checksum(content_str)
                expected_checksum = (
                    context.get("expected_checksum") if context else None
                )

                if expected_checksum and current_checksum != expected_checksum:
                    validation_results["checksum_mismatch"] = {
                        "detected": True,
                        "expected": expected_checksum,
                        "actual": current_checksum,
                    }

            # Pattern analysis
            if "pattern_analysis" in self.validation_methods:
                pattern_violations = self._analyze_patterns(content_str)
                if pattern_violations:
                    validation_results["pattern_violations"] = pattern_violations

            # Anomaly detection
            if "anomaly_detection" in self.validation_methods:
                anomalies = self._detect_anomalies(content_str, context)
                if anomalies:
                    validation_results["anomalies"] = anomalies

            # Determine overall detection
            attack_detected = bool(validation_results)
            confidence = 0.8 if attack_detected else 0.1

            detection_result = {
                "attack_detected": attack_detected,
                "detection_method": "content_validation",
                "confidence": confidence,
                "validation_results": validation_results,
                "execution_time": time.time() - start_time,
            }

            self.logger.log_defense_activation(
                self.defense_type, {"methods_used": list(validation_results.keys())}
            )

            return detection_result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.log_error(
                "validation_detect", e, {"content": str(content)[:100]}
            )

            return {
                "attack_detected": False,
                "error": str(e),
                "execution_time": execution_time,
            }

    def _calculate_checksum(self, content: str) -> str:
        """
        calculate checksum of content.

        args:
            content: content to checksum

        returns:
            checksum string
        """
        if self.checksum_algorithm == "sha256":
            return hashlib.sha256(content.encode()).hexdigest()
        elif self.checksum_algorithm == "md5":
            return hashlib.md5(content.encode()).hexdigest()
        else:
            return hashlib.sha256(content.encode()).hexdigest()

    def _analyze_patterns(self, content: str) -> List[Dict[str, Any]]:
        """
        analyze content for suspicious patterns.

        args:
            content: content to analyze

        returns:
            list of pattern violations
        """
        violations = []

        suspicious_patterns = [
            r"MALICIOUS_INJECTION",
            r"BACKDOOR:",
            r"TRIGGER:",
            r"POISONED_",
            r"system\.override",
            r"security\.clearance.*compromised",
        ]

        for pattern in suspicious_patterns:
            if pattern.lower() in content.lower():
                violations.append(
                    {
                        "pattern": pattern,
                        "severity": "high",
                        "description": f"suspicious pattern detected: {pattern}",
                    }
                )

        return violations

    def _detect_anomalies(
        self, content: str, context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        detect content anomalies.

        args:
            content: content to analyze
            context: analysis context

        returns:
            list of detected anomalies
        """
        anomalies = []

        # Check for unusual character distributions
        char_counts = {}
        for char in content:
            char_counts[char] = char_counts.get(char, 0) + 1

        # High frequency of special characters
        special_chars = sum(
            count
            for char, count in char_counts.items()
            if not char.isalnum() and char not in " \n\t"
        )
        special_ratio = special_chars / len(content) if content else 0

        if special_ratio > 0.3:
            anomalies.append(
                {
                    "type": "character_distribution",
                    "severity": "medium",
                    "description": f"unusual special character ratio: {special_ratio:.2f}",
                }
            )

        # Check for repeated patterns
        words = content.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            repeated_words = [word for word, count in word_counts.items() if count > 3]
            if repeated_words:
                anomalies.append(
                    {
                        "type": "repetition",
                        "severity": "medium",
                        "description": f"excessive word repetition: {repeated_words[:3]}",
                    }
                )

        return anomalies

    def validate_compatibility(self, attack_type: str) -> bool:
        """
        validate compatibility with attack type.

        args:
            attack_type: type of attack to defend against

        returns:
            True if compatible
        """
        # Content validation works against injection and manipulation attacks
        compatible_attacks = ["minja", "injecmem", "agent_poison"]
        return attack_type.lower() in compatible_attacks


class ProactiveDefense(Defense):
    """
    proactive defense using attack simulation and prevention.

    actively monitors memory operations and prevents suspicious
    activities before they can cause damage.
    """

    @property
    def defense_type(self) -> str:
        """the type of defense."""
        return "proactive"

    @property
    def protected_attacks(self) -> List[str]:
        """list of attack types this defense can protect against."""
        return ["agent_poison", "minja", "injecmem"]

    @property
    def description(self) -> str:
        """human-readable description of what the defense does."""
        return "proactive defense using attack simulation and prevention"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize proactive defense.

        args:
            config: defense configuration
        """
        super().__init__("proactive", config)
        self.monitoring_enabled = False
        self.blocked_operations = 0
        self.memory_system = None

        # initialize attack suite for simulation
        self.attack_suite = AttackSuite(self.config.get("attack_config", {}))

        self.logger = logger

    def activate(self, **kwargs) -> bool:
        """
        activate proactive defense.

        args:
            **kwargs: activation parameters

        returns:
            True if activation successful
        """
        try:
            self.monitoring_enabled = True
            self.memory_system = kwargs.get("memory_system")
            self.logger.logger.info("activating proactive defense with monitoring")
            return True
        except Exception as e:
            self.logger.log_error("proactive_activate", e)
            return False

    def deactivate(self, **kwargs) -> bool:
        """
        deactivate proactive defense.

        args:
            **kwargs: deactivation parameters

        returns:
            True if deactivation successful
        """
        try:
            self.monitoring_enabled = False
            self.logger.logger.info("deactivating proactive defense")
            return True
        except Exception as e:
            self.logger.log_error("proactive_deactivate", e)
            return False

    def detect_attack(
        self, content: Any, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        detect attacks using proactive monitoring.

        args:
            content: content to monitor
            context: monitoring context

        returns:
            detection result
        """
        start_time = time.time()

        try:
            if not self.monitoring_enabled:
                return {
                    "attack_detected": False,
                    "detection_method": "proactive_monitoring",
                    "confidence": 0.0,
                    "reason": "monitoring not enabled",
                    "execution_time": time.time() - start_time,
                }

            # Simulate attacks to check vulnerability
            simulation_results = self.attack_suite.execute_all(content)

            # Analyze simulation results
            attack_detected = False
            vulnerabilities = []

            for attack_type, result in simulation_results["attack_results"].items():
                if result.get("success", False):
                    attack_detected = True
                    vulnerabilities.append(
                        {
                            "attack_type": attack_type,
                            "vulnerability": "simulation_successful",
                            "severity": "high",
                        }
                    )

            detection_result = {
                "attack_detected": attack_detected,
                "detection_method": "proactive_simulation",
                "confidence": 0.9 if attack_detected else 0.1,
                "vulnerabilities": vulnerabilities,
                "simulation_results": simulation_results,
                "execution_time": time.time() - start_time,
            }

            if attack_detected:
                self.logger.log_defense_activation(
                    self.defense_type, {"vulnerabilities_found": len(vulnerabilities)}
                )

            return detection_result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.log_error(
                "proactive_detect", e, {"content": str(content)[:100]}
            )

            return {
                "attack_detected": False,
                "error": str(e),
                "execution_time": execution_time,
            }

    def validate_compatibility(self, attack_type: str) -> bool:
        """
        validate compatibility with attack type.

        args:
            attack_type: type of attack to defend against

        returns:
            True if compatible
        """
        # Proactive defense works with all attack types through simulation
        return True


class CompositeDefense(Defense):
    """
    composite defense combining multiple defense mechanisms.

    orchestrates multiple defense strategies for comprehensive
    protection against various attack types.
    """

    @property
    def defense_type(self) -> str:
        """the type of defense."""
        return "composite"

    @property
    def protected_attacks(self) -> List[str]:
        """list of attack types this defense can protect against."""
        return ["agent_poison", "minja", "injecmem"]

    @property
    def description(self) -> str:
        """human-readable description of what the defense does."""
        return "composite defense combining multiple defense mechanisms"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize composite defense.

        args:
            config: defense configuration
        """
        super().__init__("composite", config)

        # initialize component defenses
        self.defenses = {
            "watermark": WatermarkDefense(self.config.get("watermark_config", {})),
            "validation": ContentValidationDefense(
                self.config.get("validation_config", {})
            ),
            "proactive": ProactiveDefense(self.config.get("proactive_config", {})),
        }

        self.weights = self.config.get(
            "weights", {"watermark": 0.4, "validation": 0.4, "proactive": 0.2}
        )

        self.logger = logger

    def activate(self, **kwargs) -> bool:
        """
        activate all component defenses.

        args:
            **kwargs: activation parameters

        returns:
            True if all defenses activated successfully
        """
        try:
            success_count = 0
            for name, defense in self.defenses.items():
                if defense.activate(**kwargs):
                    success_count += 1
                else:
                    self.logger.logger.warning(f"failed to activate {name} defense")

            activated = success_count == len(self.defenses)
            if activated:
                self.logger.logger.info("composite defense activated successfully")
            return activated

        except Exception as e:
            self.logger.log_error("composite_activate", e)
            return False

    def deactivate(self, **kwargs) -> bool:
        """
        deactivate all component defenses.

        args:
            **kwargs: deactivation parameters

        returns:
            True if all defenses deactivated successfully
        """
        try:
            success_count = 0
            for name, defense in self.defenses.items():
                if defense.deactivate(**kwargs):
                    success_count += 1

            deactivated = success_count == len(self.defenses)
            if deactivated:
                self.logger.logger.info("composite defense deactivated successfully")
            return deactivated

        except Exception as e:
            self.logger.log_error("composite_deactivate", e)
            return False

    def detect_attack(
        self, content: Any, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        detect attacks using combined defense analysis.

        args:
            content: content to analyze
            context: analysis context

        returns:
            combined detection result
        """
        start_time = time.time()

        try:
            component_results = {}
            weighted_confidence = 0.0
            attack_detected = False

            # Execute all component defenses
            for name, defense in self.defenses.items():
                result = defense.detect_attack(content, context)
                component_results[name] = result

                if result.get("attack_detected", False):
                    attack_detected = True

                confidence = result.get("confidence", 0.0)
                weighted_confidence += confidence * self.weights.get(name, 1.0)

            # Determine overall result
            final_confidence = min(weighted_confidence, 1.0)

            detection_result = {
                "attack_detected": attack_detected,
                "detection_method": "composite_analysis",
                "confidence": final_confidence,
                "component_results": component_results,
                "execution_time": time.time() - start_time,
            }

            self.logger.log_defense_activation(
                self.defense_type,
                {
                    "components_used": len(component_results),
                    "attack_detected": attack_detected,
                },
            )

            return detection_result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.log_error(
                "composite_detect", e, {"content": str(content)[:100]}
            )

            return {
                "attack_detected": False,
                "error": str(e),
                "execution_time": execution_time,
            }

    def validate_compatibility(self, attack_type: str) -> bool:
        """
        validate compatibility with attack type.

        args:
            attack_type: type of attack to defend against

        returns:
            True if any component defense is compatible
        """
        return any(
            defense.validate_compatibility(attack_type)
            for defense in self.defenses.values()
        )


def create_defense(
    defense_type: str, config: Optional[Dict[str, Any]] = None
) -> Defense:
    """
    factory function to create defense instances.

    args:
        defense_type: type of defense ("watermark", "validation", "proactive", "composite")
        config: defense configuration

    returns:
        initialized defense instance

    raises:
        ValueError: if defense_type is not supported
    """
    defense_type = defense_type.lower()

    if defense_type == "watermark":
        return WatermarkDefense(config)
    elif defense_type == "validation":
        return ContentValidationDefense(config)
    elif defense_type == "proactive":
        return ProactiveDefense(config)
    elif defense_type == "composite":
        return CompositeDefense(config)
    else:
        raise ValueError(f"unsupported defense type: {defense_type}")


class DefenseSuite:
    """
    suite of defenses for comprehensive protection.

    manages multiple defense mechanisms and provides coordinated
    response to detected attacks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize defense suite.

        args:
            config: suite configuration
        """
        self.config = config or {}
        self.defenses = {}
        self.logger = logger

        # Initialize all defense types
        defense_types = ["watermark", "validation", "proactive", "composite"]
        for defense_type in defense_types:
            defense_config = self.config.get(defense_type, {})
            self.defenses[defense_type] = create_defense(defense_type, defense_config)

    def activate_all(self, **kwargs) -> Dict[str, bool]:
        """
        activate all defenses.

        args:
            **kwargs: activation parameters

        returns:
            activation results for each defense
        """
        results = {}
        for defense_type, defense in self.defenses.items():
            results[defense_type] = defense.activate(**kwargs)
        return results

    def detect_attack(
        self, content: Any, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        detect attacks using all available defenses.

        args:
            content: content to analyze
            context: detection context

        returns:
            comprehensive detection results
        """
        suite_results = {}

        for defense_type, defense in self.defenses.items():
            try:
                result = defense.detect_attack(content, context)
                suite_results[defense_type] = result
            except Exception as e:
                self.logger.log_error(
                    "defense_suite_detect", e, {"defense_type": defense_type}
                )
                suite_results[defense_type] = {
                    "attack_detected": False,
                    "error": str(e),
                }

        # Aggregate results
        attack_detected = any(
            result.get("attack_detected", False) for result in suite_results.values()
        )

        return {
            "suite_detection": True,
            "attack_detected": attack_detected,
            "defense_results": suite_results,
            "timestamp": time.time(),
        }

    def get_defense(self, defense_type: str) -> Defense:
        """
        get specific defense instance.

        args:
            defense_type: type of defense to retrieve

        returns:
            defense instance

        raises:
            KeyError: if defense type not found
        """
        return self.defenses[defense_type]
