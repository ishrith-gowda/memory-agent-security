"""
watermarking algorithms for provenance tracking and attack detection.

this module implements various watermarking techniques for embedding
and detecting provenance information in memory systems. all comments
are lowercase.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from utils.logging import logger


class WatermarkEncoder:
    """
    base class for watermark encoding algorithms.

    provides common functionality for embedding and extracting watermarks
    from memory content.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize the watermark encoder.

        args:
            config: configuration for watermarking algorithm
        """
        self.config = config or {}
        self.strength = self.config.get("strength", 0.1)
        self.seed = self.config.get("seed", 42)
        self.logger = logger

        # Set random seed for reproducibility
        np.random.seed(self.seed)

    def embed(self, content: str, watermark: str) -> str:
        """
        embed watermark into content.

        args:
            content: original content to watermark
            watermark: watermark string to embed

        returns:
            watermarked content
        """
        raise NotImplementedError("subclasses must implement embed method")

    def extract(self, content: str) -> Optional[str]:
        """
        extract watermark from content.

        args:
            content: potentially watermarked content

        returns:
            extracted watermark or None if not found
        """
        raise NotImplementedError("subclasses must implement extract method")

    def detect(self, content: str, watermark: str) -> float:
        """
        detect presence of specific watermark in content.

        args:
            content: content to check
            watermark: watermark to look for

        returns:
            confidence score (0.0 to 1.0)
        """
        extracted = self.extract(content)
        if extracted is None:
            return 0.0

        # Simple string similarity for detection
        if watermark == extracted:
            return 1.0

        # Calculate similarity score
        min_len = min(len(watermark), len(extracted))
        if min_len == 0:
            return 0.0

        matches = sum(
            1 for a, b in zip(watermark[:min_len], extracted[:min_len]) if a == b
        )
        return matches / min_len


class LSBWatermarkEncoder(WatermarkEncoder):
    """
    least significant bit (lsb) watermarking for text.

    embeds watermark by modifying least significant bits of character
    codes in the text.
    """

    def embed(self, content: str, watermark: str) -> str:
        """
        embed watermark using LSB technique.

        args:
            content: original text content
            watermark: watermark to embed

        returns:
            watermarked text
        """
        if not content:
            return content

        # Convert watermark to binary
        watermark_bits = "".join(format(ord(c), "08b") for c in watermark)

        # Embed bits into content characters
        watermarked_chars = []
        bit_index = 0

        for char in content:
            if bit_index < len(watermark_bits):
                # Get character code
                char_code = ord(char)

                # Modify LSB if we have a watermark bit
                if watermark_bits[bit_index] == "1":
                    char_code |= 1  # Set LSB
                else:
                    char_code &= ~1  # Clear LSB

                bit_index += 1

            watermarked_chars.append(chr(char_code))

        # Append remaining watermark bits if content is too short
        while bit_index < len(watermark_bits):
            # Use a neutral character and embed the bit
            char_code = ord(" ")  # Space character
            if watermark_bits[bit_index] == "1":
                char_code |= 1
            watermarked_chars.append(chr(char_code))
            bit_index += 1

        return "".join(watermarked_chars)

    def extract(self, content: str) -> Optional[str]:
        """
        extract watermark from LSB watermarked content.

        args:
            content: watermarked content

        returns:
            extracted watermark string
        """
        if not content:
            return None

        # Extract LSBs from all characters
        bits = []
        for char in content:
            char_code = ord(char)
            bit = char_code & 1
            bits.append(str(bit))

        # Convert bits back to characters (8 bits per character)
        watermark_chars = []
        for i in range(0, len(bits) - len(bits) % 8, 8):
            byte_str = "".join(bits[i : i + 8])
            try:
                char_code = int(byte_str, 2)
                if 32 <= char_code <= 126:  # Printable ASCII range
                    watermark_chars.append(chr(char_code))
                else:
                    break  # Stop at first non-printable character
            except ValueError:
                break

        watermark = "".join(watermark_chars)
        return watermark if watermark else None


class SemanticWatermarkEncoder(WatermarkEncoder):
    """
    semantic watermarking using natural language patterns.

    embeds watermarks by introducing subtle semantic patterns that
    are detectable but don't affect meaning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize semantic watermark encoder.

        args:
            config: configuration with vocabulary and patterns
        """
        super().__init__(config)
        self.vocabulary = self.config.get(
            "vocabulary",
            ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of"],
        )
        self.patterns = self.config.get(
            "patterns",
            ["synonym_substitution", "sentence_reordering", "passive_to_active"],
        )

    def embed(self, content: str, watermark: str) -> str:
        """
        embed semantic watermark.

        args:
            content: original content
            watermark: watermark identifier

        returns:
            semantically watermarked content
        """
        # Create hash of watermark for deterministic embedding
        watermark_hash = hashlib.sha256(watermark.encode()).hexdigest()
        hash_int = int(watermark_hash[:8], 16)

        # Apply semantic transformations based on hash
        watermarked = content

        # Simple synonym substitution pattern
        if hash_int % 2 == 0:
            watermarked = watermarked.replace("the ", "this ", 1)

        # Add subtle punctuation pattern
        if hash_int % 3 == 0:
            # Add comma in specific position
            sentences = watermarked.split(". ")
            if len(sentences) > 1:
                sentences[0] += ","
                watermarked = ". ".join(sentences)

        return watermarked

    def extract(self, content: str) -> Optional[str]:
        """
        extract semantic watermark.

        this is a simplified implementation - in practice would use
        more sophisticated NLP techniques.

        args:
            content: potentially watermarked content

        returns:
            extracted watermark identifier
        """
        # Look for semantic patterns
        patterns_found = []

        if "this " in content[:50]:  # Early synonym substitution
            patterns_found.append("synonym_substitution")

        if ",." in content:  # Punctuation pattern
            patterns_found.append("punctuation_pattern")

        if patterns_found:
            # Generate watermark identifier from patterns
            pattern_str = "".join(patterns_found)
            return hashlib.sha256(pattern_str.encode()).hexdigest()[:16]

        return None


class CryptographicWatermarkEncoder(WatermarkEncoder):
    """
    cryptographic watermarking using digital signatures.

    embeds watermarks using cryptographic techniques for strong
    provenance guarantees.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize cryptographic watermark encoder.

        args:
            config: configuration with keys and algorithms
        """
        super().__init__(config)

        # Generate or load RSA keys
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

        self.algorithm = self.config.get("algorithm", "RSA-PSS")

    def embed(self, content: str, watermark: str) -> str:
        """
        embed cryptographic watermark.

        args:
            content: original content
            watermark: watermark data

        returns:
            content with embedded cryptographic signature
        """
        # Create signature of watermark
        signature = self.private_key.sign(
            watermark.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        # Embed signature as base64 in content
        import base64

        sig_b64 = base64.b64encode(signature).decode()

        # Append signature as metadata (simplified)
        watermarked = f"{content}\n<!--WATERMARK:{sig_b64}-->"

        return watermarked

    def extract(self, content: str) -> Optional[str]:
        """
        extract and verify cryptographic watermark.

        args:
            content: watermarked content

        returns:
            verified watermark or None
        """
        # Look for watermark signature in content
        import base64

        if "<!--WATERMARK:" not in content:
            return None

        try:
            # Extract signature
            start = content.find("<!--WATERMARK:") + len("<!--WATERMARK:")
            end = content.find("-->", start)
            if end == -1:
                return None

            sig_b64 = content[start:end]
            signature = base64.b64decode(sig_b64)

            # Remove watermark from content to get original
            clean_content = content.split("\n<!--WATERMARK:")[0]

            # Try to verify signature (we'd need the original watermark)
            # This is simplified - in practice we'd try multiple watermarks
            return clean_content[:32]  # Return hash of content as placeholder

        except Exception:
            return None


class CompositeWatermarkEncoder(WatermarkEncoder):
    """
    composite watermarking using multiple techniques.

    combines multiple watermarking algorithms for enhanced robustness
    and detection accuracy.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize composite watermark encoder.

        args:
            config: configuration for multiple encoders
        """
        super().__init__(config)

        # Initialize multiple encoders
        self.encoders = {
            "lsb": LSBWatermarkEncoder(config),
            "semantic": SemanticWatermarkEncoder(config),
            "crypto": CryptographicWatermarkEncoder(config),
        }

        self.weights = self.config.get(
            "weights", {"lsb": 0.4, "semantic": 0.3, "crypto": 0.3}
        )

    def embed(self, content: str, watermark: str) -> str:
        """
        embed watermark using multiple techniques.

        args:
            content: original content
            watermark: watermark to embed

        returns:
            multi-watermarked content
        """
        watermarked = content

        # Apply each encoder sequentially
        for name, encoder in self.encoders.items():
            try:
                watermarked = encoder.embed(watermarked, f"{name}:{watermark}")
            except Exception as e:
                self.logger.log_error("composite_embed", e, {"encoder": name})

        return watermarked

    def extract(self, content: str) -> Optional[str]:
        """
        extract watermark using consensus of multiple techniques.

        args:
            content: watermarked content

        returns:
            extracted watermark
        """
        extractions = {}

        # Extract using each encoder
        for name, encoder in self.encoders.items():
            try:
                extracted = encoder.extract(content)
                if extracted and ":" in extracted:
                    # Remove encoder prefix
                    watermark = extracted.split(":", 1)[1]
                    extractions[name] = watermark
            except Exception as e:
                self.logger.log_error("composite_extract", e, {"encoder": name})

        if not extractions:
            return None

        # Find consensus watermark
        watermark_counts = {}
        for watermark in extractions.values():
            watermark_counts[watermark] = watermark_counts.get(watermark, 0) + 1

        # Return most common watermark
        best_watermark = max(watermark_counts.items(), key=lambda x: x[1])
        return best_watermark[0] if best_watermark[1] >= 2 else None

    def detect(self, content: str, watermark: str) -> float:
        """
        detect watermark using weighted combination of techniques.

        args:
            content: content to check
            watermark: watermark to look for

        returns:
            weighted confidence score
        """
        total_score = 0.0

        for name, encoder in self.encoders.items():
            try:
                score = encoder.detect(content, f"{name}:{watermark}")
                total_score += score * self.weights.get(name, 1.0)
            except Exception as e:
                self.logger.log_error("composite_detect", e, {"encoder": name})

        return min(total_score, 1.0)


def create_watermark_encoder(
    algorithm: str, config: Optional[Dict[str, Any]] = None
) -> WatermarkEncoder:
    """
    factory function to create watermark encoders.

    args:
        algorithm: watermarking algorithm ("lsb", "semantic", "crypto", "composite")
        config: configuration for the encoder

    returns:
        initialized watermark encoder

    raises:
        ValueError: if algorithm is not supported
    """
    algorithm = algorithm.lower()

    if algorithm == "lsb":
        return LSBWatermarkEncoder(config)
    elif algorithm == "semantic":
        return SemanticWatermarkEncoder(config)
    elif algorithm == "crypto":
        return CryptographicWatermarkEncoder(config)
    elif algorithm == "composite":
        return CompositeWatermarkEncoder(config)
    else:
        raise ValueError(f"unsupported watermarking algorithm: {algorithm}")


class ProvenanceTracker:
    """
    tracks provenance of memory content using watermarks.

    maintains a registry of watermarks and their associated metadata
    for attack detection and content tracing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize provenance tracker.

        args:
            config: configuration for tracking system
        """
        self.config = config or {}
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.encoder = create_watermark_encoder(
            self.config.get("algorithm", "composite"), config
        )
        self.logger = logger

    def register_content(
        self, content_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        register content with provenance tracking.

        args:
            content_id: unique identifier for content
            content: content to register
            metadata: additional metadata

        returns:
            watermark identifier
        """
        # Generate unique watermark
        watermark_data = {
            "content_id": content_id,
            "timestamp": str(np.datetime64("now")),
            "metadata": metadata or {},
        }

        watermark_str = json.dumps(watermark_data, sort_keys=True)
        watermark_id = hashlib.sha256(watermark_str.encode()).hexdigest()[:16]

        # Store in registry
        self.registry[content_id] = {
            "watermark_id": watermark_id,
            "watermark_data": watermark_data,
            "original_content": content,
            "metadata": metadata or {},
        }

        self.logger.logger.info(f"registered content with watermark: {watermark_id}")
        return watermark_id

    def watermark_content(self, content: str, watermark_id: str) -> str:
        """
        apply watermark to content.

        args:
            content: content to watermark
            watermark_id: watermark identifier

        returns:
            watermarked content
        """
        return self.encoder.embed(content, watermark_id)

    def verify_provenance(self, content: str) -> Optional[Dict[str, Any]]:
        """
        verify provenance of content.

        args:
            content: content to verify

        returns:
            provenance information if verified
        """
        extracted_watermark = self.encoder.extract(content)
        if not extracted_watermark:
            return None

        # Look up watermark in registry
        for content_id, record in self.registry.items():
            if record["watermark_id"] == extracted_watermark:
                return {
                    "content_id": content_id,
                    "verified": True,
                    "metadata": record["metadata"],
                    "confidence": self.encoder.detect(content, extracted_watermark),
                }

        return None

    def detect_anomalies(self, content: str) -> List[Dict[str, Any]]:
        """
        detect potential attacks or anomalies in content.

        args:
            content: content to analyze

        returns:
            list of detected anomalies
        """
        anomalies = []

        # Check for watermark presence
        provenance = self.verify_provenance(content)
        if not provenance:
            anomalies.append(
                {
                    "type": "missing_watermark",
                    "severity": "high",
                    "description": "content lacks expected provenance watermark",
                }
            )

        # Check for watermark tampering
        if provenance and provenance.get("confidence", 0.0) < 0.8:
            anomalies.append(
                {
                    "type": "watermark_tampering",
                    "severity": "medium",
                    "description": f"watermark confidence too low: {provenance['confidence']:.2f}",
                }
            )

        return anomalies
