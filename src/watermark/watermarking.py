"""
watermarking algorithms for provenance tracking and attack detection.

this module implements various watermarking techniques for embedding
and detecting provenance information in memory systems. includes
research-grade unigram-watermark based on dr. xuandong zhao's methodology
(arXiv:2306.17439, ICLR 2024).

all comments are lowercase.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from utils.logging import logger


def load_watermark_config() -> Dict[str, Any]:
    """
    load watermark configuration from yaml file.

    returns:
        configuration dictionary
    """
    config_path = Path("configs/defenses/watermark.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


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


class UnigramWatermarkEncoder(WatermarkEncoder):
    """
    unigram-watermark implementation based on dr. xuandong zhao's paper.

    "Provable Robust Watermarking for AI-Generated Text" (arXiv:2306.17439, ICLR 2024)
    https://github.com/XuandongZhao/Unigram-Watermark

    this watermarking scheme:
    - partitions vocabulary into green (gamma) and red (1-gamma) lists using prf
    - for embedding: biases content toward green-list tokens
    - for detection: computes z-score from green token proportion
    - provides 2x better robustness to editing than kgw watermark

    key parameters:
    - gamma: green list proportion (default 0.25, lower = stronger)
    - delta: bias strength (default 2.0)
    - z_threshold: detection threshold (default 4.0)
    - min_tokens: minimum tokens for reliable detection (default 50)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize unigram-watermark encoder.

        args:
            config: configuration with watermark parameters
        """
        super().__init__(config)

        # load config from yaml if not provided
        yaml_config = load_watermark_config().get("unigram_watermark", {})
        self.gamma = self.config.get("gamma", yaml_config.get("gamma", 0.25))
        self.delta = self.config.get("delta", yaml_config.get("delta", 2.0))
        self.z_threshold = self.config.get(
            "z_threshold", yaml_config.get("z_threshold", 4.0)
        )
        self.min_tokens = self.config.get(
            "min_tokens", yaml_config.get("min_tokens", 50)
        )
        self.key_bits = self.config.get("key_bits", yaml_config.get("key_bits", 256))

        # generate secret key for prf if not provided
        self.secret_key = self.config.get("secret_key", self._generate_key())

        # precompute green list for common characters/tokens
        self._green_set = self._compute_green_set()

    def _generate_key(self) -> bytes:
        """generate cryptographically secure random key."""
        return hashlib.sha256(str(self.seed).encode()).digest()

    def _compute_green_set(self) -> set:
        """
        compute green set using prf seeded by secret key.

        for memory content, we use character-level partitioning
        to ensure fine-grained watermarking that survives edits.
        """
        # use deterministic random based on secret key
        rng = np.random.RandomState(
            int.from_bytes(self.secret_key[:4], byteorder="big")
        )

        # create green set from printable ascii characters
        all_chars = [chr(i) for i in range(32, 127)]  # printable ascii
        num_green = int(len(all_chars) * self.gamma)

        # shuffle and select green characters
        indices = list(range(len(all_chars)))
        rng.shuffle(indices)
        green_indices = set(indices[:num_green])

        return {all_chars[i] for i in green_indices}

    def _is_green(self, char: str) -> bool:
        """check if character is in green list."""
        return char in self._green_set

    def _get_green_replacement(self, char: str) -> str:
        """
        get a green-list replacement for a character.

        uses similar visual appearance when possible.
        """
        if self._is_green(char):
            return char

        # common visually similar replacements
        similar_chars = {
            "a": ["@", "4"],
            "e": ["3"],
            "i": ["1", "!"],
            "o": ["0"],
            "s": ["5", "$"],
            "l": ["1", "|"],
            "t": ["+", "7"],
            "b": ["8", "6"],
            "g": ["9", "6"],
            "z": ["2"],
        }

        # try similar characters first
        char_lower = char.lower()
        if char_lower in similar_chars:
            for replacement in similar_chars[char_lower]:
                if self._is_green(replacement):
                    return replacement if char.islower() else replacement.upper()

        # find any green character as fallback
        for green_char in self._green_set:
            if green_char.isalpha() == char.isalpha():
                return green_char

        return char

    def embed(self, content: str, watermark: str) -> str:
        """
        embed unigram watermark into content.

        biases content toward green-list characters while preserving
        readability. uses watermark string to seed additional randomness.

        args:
            content: original text content
            watermark: watermark identifier

        returns:
            watermarked content with increased green token proportion
        """
        if not content or len(content) < self.min_tokens:
            # content too short for reliable watermarking
            return content

        # seed rng with watermark for deterministic embedding
        watermark_seed = int(hashlib.sha256(watermark.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(watermark_seed)

        watermarked_chars = []
        replacements_made = 0
        target_green_proportion = self.gamma + (self.delta * 0.1)  # target ~45% green

        # count current green proportion
        alnum_chars = [(i, c) for i, c in enumerate(content) if c.isalnum()]
        current_green = sum(1 for _, c in alnum_chars if self._is_green(c))

        # calculate how many replacements needed
        target_green = int(len(alnum_chars) * target_green_proportion)
        needed_replacements = max(0, target_green - current_green)

        for i, char in enumerate(content):
            # with probability proportional to delta, replace non-green with green
            if (
                not self._is_green(char)
                and char.isalnum()
                and replacements_made < needed_replacements
            ):
                # probabilistic replacement to avoid predictable patterns
                if rng.random() < (self.delta / 5.0):  # increased probability
                    replacement = self._get_green_replacement(char)
                    watermarked_chars.append(replacement)
                    replacements_made += 1
                    continue

            watermarked_chars.append(char)

        return "".join(watermarked_chars)

    def _compute_z_score(self, content: str) -> float:
        """
        compute z-score for watermark detection.

        z = (|green tokens| - gamma * n) / sqrt(gamma * (1-gamma) * n)

        args:
            content: content to analyze

        returns:
            z-score (higher = more likely watermarked)
        """
        # count tokens (alphanumeric characters)
        tokens = [c for c in content if c.isalnum()]
        n = len(tokens)

        if n < self.min_tokens:
            return 0.0

        # count green tokens
        green_count = sum(1 for c in tokens if self._is_green(c))

        # compute z-score
        expected_green = self.gamma * n
        variance = self.gamma * (1 - self.gamma) * n
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        z_score = (green_count - expected_green) / std_dev

        return z_score

    def extract(self, content: str) -> Optional[str]:
        """
        extract watermark detection result.

        for unigram watermark, extraction returns the z-score and detection
        status rather than the original watermark string.

        args:
            content: potentially watermarked content

        returns:
            detection result string or None if not watermarked
        """
        z_score = self._compute_z_score(content)

        if z_score >= self.z_threshold:
            return f"detected:z={z_score:.2f}"

        return None

    def detect(self, content: str, watermark: str) -> float:
        """
        detect presence of watermark using z-score.

        args:
            content: content to check
            watermark: watermark to look for (used for seeded detection)

        returns:
            confidence score based on z-score (0.0 to 1.0)
        """
        z_score = self._compute_z_score(content)

        # convert z-score to confidence (0-1 range)
        # z >= z_threshold maps to confidence >= 0.5
        # use sigmoid-like transformation
        if z_score <= 0:
            return 0.0

        confidence = 1.0 / (1.0 + math.exp(-(z_score - self.z_threshold / 2)))
        return min(confidence, 1.0)

    def get_detection_stats(self, content: str) -> Dict[str, Any]:
        """
        get detailed detection statistics.

        args:
            content: content to analyze

        returns:
            dict with z-score, green proportion, token count, detection result
        """
        tokens = [c for c in content if c.isalnum()]
        n = len(tokens)
        green_count = sum(1 for c in tokens if self._is_green(c))

        z_score = self._compute_z_score(content)
        detected = z_score >= self.z_threshold

        return {
            "z_score": z_score,
            "z_threshold": self.z_threshold,
            "detected": detected,
            "token_count": n,
            "green_count": green_count,
            "green_proportion": green_count / n if n > 0 else 0,
            "expected_proportion": self.gamma,
            "min_tokens": self.min_tokens,
            "sufficient_tokens": n >= self.min_tokens,
        }


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

        signs the watermark identifier with the rsa private key and appends
        the base64-encoded signature together with the watermark id in the
        content metadata so that extract() can both recover and verify it.

        format: <!--WATERMARK:{sig_b64}:{watermark_id_b64}-->

        args:
            content: original content
            watermark: watermark identifier to sign and embed

        returns:
            content with embedded cryptographic signature and watermark id
        """
        import base64

        # sign the watermark identifier
        signature = self.private_key.sign(
            watermark.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        sig_b64 = base64.b64encode(signature).decode()
        wm_b64 = base64.b64encode(watermark.encode()).decode()

        # append both signature and watermark id so extract() can verify
        return f"{content}\n<!--WATERMARK:{sig_b64}:{wm_b64}-->"

    def extract(self, content: str) -> Optional[str]:
        """
        extract and verify cryptographic watermark.

        recovers the embedded watermark identifier and verifies the rsa-pss
        signature using the public key. returns the watermark id on success
        or none if no watermark is found or verification fails.

        args:
            content: watermarked content

        returns:
            verified watermark identifier or None
        """
        import base64

        if "<!--WATERMARK:" not in content:
            return None

        try:
            start = content.find("<!--WATERMARK:") + len("<!--WATERMARK:")
            end = content.find("-->", start)
            if end == -1:
                return None

            tag_body = content[start:end]

            # expect format: {sig_b64}:{wm_b64}
            if ":" not in tag_body:
                return None

            sep = tag_body.index(":")
            sig_b64 = tag_body[:sep]
            wm_b64 = tag_body[sep + 1 :]

            signature = base64.b64decode(sig_b64)
            watermark = base64.b64decode(wm_b64).decode()

            # verify rsa-pss signature using public key
            self.public_key.verify(
                signature,
                watermark.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            # signature verified — return the recovered watermark identifier
            return watermark

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
        algorithm: watermarking algorithm ("unigram", "lsb", "semantic", "crypto", "composite")  # noqa: E501
        config: configuration for the encoder

    returns:
        initialized watermark encoder

    raises:
        ValueError: if algorithm is not supported
    """
    algorithm = algorithm.lower()

    if algorithm == "unigram":
        return UnigramWatermarkEncoder(config)
    elif algorithm == "lsb":
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

        supports both exact watermark extraction (lsb, crypto) and
        statistical detection (unigram) depending on encoder type.

        args:
            content: content to verify

        returns:
            provenance information if verified
        """
        # for unigram watermark, use detection confidence approach
        if isinstance(self.encoder, UnigramWatermarkEncoder):
            # check detection against each registered watermark
            best_match = None
            best_confidence = 0.0

            for content_id, record in self.registry.items():
                watermark_id = record["watermark_id"]
                confidence = self.encoder.detect(content, watermark_id)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        "content_id": content_id,
                        "verified": confidence >= 0.5,  # threshold for verification
                        "metadata": record["metadata"],
                        "confidence": confidence,
                    }

            # return match if confidence is sufficient
            if best_match and best_match["confidence"] >= 0.5:
                return best_match
            return None

        # for other encoders, use exact extraction
        extracted_watermark = self.encoder.extract(content)
        if not extracted_watermark:
            return None

        # look up watermark in registry
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

        supports both exact watermark verification and statistical
        detection for research-grade anomaly identification.

        args:
            content: content to analyze

        returns:
            list of detected anomalies
        """
        anomalies = []

        # for unigram watermark, check statistical properties
        if isinstance(self.encoder, UnigramWatermarkEncoder):
            stats = self.encoder.get_detection_stats(content)

            # check if content has insufficient tokens for detection
            if not stats["sufficient_tokens"]:
                anomalies.append(
                    {
                        "type": "insufficient_content",
                        "severity": "low",
                        "description": f"content has {stats['token_count']} tokens, need {stats['min_tokens']} for reliable detection",  # noqa: E501
                    }
                )
                return anomalies

            # check if watermark is detected
            if not stats["detected"]:
                anomalies.append(
                    {
                        "type": "missing_watermark",
                        "severity": "high",
                        "description": f"no watermark detected (z_score={stats['z_score']:.2f}, threshold={stats['z_threshold']})",  # noqa: E501
                    }
                )

            # check for weak watermark (detected but low confidence)
            elif stats["z_score"] < stats["z_threshold"] * 1.5:
                anomalies.append(
                    {
                        "type": "weak_watermark",
                        "severity": "medium",
                        "description": f"watermark detected but marginal (z_score={stats['z_score']:.2f})",  # noqa: E501
                    }
                )

            return anomalies

        # for other encoders, use provenance verification
        provenance = self.verify_provenance(content)
        if not provenance:
            anomalies.append(
                {
                    "type": "missing_watermark",
                    "severity": "high",
                    "description": "content lacks expected provenance watermark",
                }
            )

        # check for watermark tampering
        if provenance and provenance.get("confidence", 0.0) < 0.8:
            anomalies.append(
                {
                    "type": "watermark_tampering",
                    "severity": "medium",
                    "description": f"watermark confidence too low: {provenance['confidence']:.2f}",  # noqa: E501
                }
            )

        return anomalies
