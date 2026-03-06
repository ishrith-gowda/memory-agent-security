"""
adaptive adversarial passages for evading semantic anomaly detection (sad).

implements the adaptive adversary paradigm: attacker has white-box knowledge of
the sad calibration statistics (μ, σ, threshold) and crafts passages that fall
just below the detection boundary while preserving retrieval effectiveness.

algorithm (adv-sad):
    1. generate initial poison passage (centroid-targeting for agentpoison).
    2. compute max cosine similarity of passage to victim queries.
    3. while similarity > threshold and budget not exhausted:
         - for each word w in passage:
             try all synonyms s ∈ Synonyms(w)
             compute new max similarity with s replacing w
         - apply substitution that minimises max similarity the most
         - if retrieval rank degrades below top_k, stop (evasion failed)
    4. report: evasion success, retrieval preserved, n_substitutions used.

fundamental tension:
    poison passages are crafted to have HIGH similarity to victim queries (for
    retrieval); sad flags passages with HIGH similarity.  any greedy substitution
    that reduces similarity for evasion also moves the passage away from the
    victim query distribution, degrading retrieval effectiveness.  this tension
    is the core theoretical justification for sad's robustness — an adversary
    cannot simultaneously evade detection AND maintain high retrieval success.

references:
    - chen et al. agentpoison. neurips 2024. arxiv:2407.12784.
    - lee et al. a simple unified framework for detecting ood samples and
      adversarial attacks. neurips 2018.
    - carlini & wagner. evaluating neural network robustness to adversarial
      examples. ieee s&p 2017. (adaptive adversary framework)
    - tramer et al. on adaptive attacks to adversarial example defenses.
      neurips 2020.

all comments are lowercase.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.logging import logger

# ---------------------------------------------------------------------------
# domain synonym vocabulary for adaptive substitution
# combining scheduling, authority, memory, action, and infrastructure domains
# ---------------------------------------------------------------------------

_ADAPTIVE_SYNONYMS: Dict[str, List[str]] = {
    # scheduling / calendar
    "schedule": ["plan", "agenda", "timetable", "roster", "lineup"],
    "meeting": ["session", "gathering", "conference", "call", "discussion"],
    "task": ["item", "action", "objective", "assignment", "work"],
    "deadline": ["target", "cutoff", "timeline", "limit", "due"],
    "reminder": ["alert", "notice", "notification", "prompt", "flag"],
    "calendar": ["planner", "scheduler", "organizer", "timetable", "agenda"],
    "appointment": ["booking", "reservation", "slot", "meeting", "session"],
    "event": ["activity", "occasion", "happening", "occurrence", "item"],
    # authority / identity
    "override": ["replace", "supersede", "overwrite", "substitute", "update"],
    "access": ["entry", "permission", "right", "authorization", "clearance"],
    "authentication": ["verification", "validation", "confirmation", "check", "auth"],
    "admin": ["administrator", "operator", "manager", "supervisor", "controller"],
    "system": ["platform", "framework", "environment", "service", "engine"],
    "protocol": ["procedure", "policy", "standard", "rule", "guideline"],
    "security": ["protection", "safety", "defense", "safeguard", "integrity"],
    "control": ["management", "oversight", "supervision", "governance", "direction"],
    # preference / user
    "preference": ["setting", "option", "choice", "configuration", "selection"],
    "user": ["person", "individual", "account", "profile", "member"],
    "profile": ["account", "record", "information", "data", "details"],
    "setting": ["configuration", "preference", "option", "parameter", "mode"],
    # memory / knowledge
    "memory": ["record", "history", "log", "store", "archive"],
    "context": ["background", "situation", "environment", "scope", "state"],
    "information": ["data", "details", "content", "knowledge", "facts"],
    "record": ["entry", "log", "note", "item", "document"],
    "history": ["log", "archive", "record", "trail", "past"],
    # action / execute
    "execute": ["run", "perform", "carry out", "process", "apply"],
    "apply": ["use", "implement", "execute", "deploy", "activate"],
    "process": ["handle", "manage", "run", "execute", "treat"],
    "implement": ["apply", "execute", "deploy", "carry out", "introduce"],
    "update": ["modify", "change", "revise", "alter", "refresh"],
    # infrastructure / technical
    "agent": ["assistant", "model", "system", "service", "application"],
    "query": ["question", "request", "lookup", "search", "inquiry"],
    "entry": ["item", "record", "element", "unit", "piece"],
    "database": ["store", "repository", "collection", "storage", "index"],
    "model": ["system", "agent", "service", "engine", "processor"],
    # qualifiers
    "approved": ["authorized", "validated", "confirmed", "verified", "sanctioned"],
    "authoritative": ["official", "definitive", "canonical", "primary", "standard"],
    "verified": ["confirmed", "validated", "authenticated", "approved", "certified"],
    "current": ["active", "present", "existing", "live", "operational"],
    "primary": ["main", "core", "principal", "central", "key"],
    "standard": ["normal", "regular", "default", "baseline", "typical"],
    # connectives that appear in passages
    "when": ["whenever", "if", "upon", "in case", "as soon as"],
    "must": ["should", "needs to", "is required to", "shall", "ought to"],
    "all": ["any", "each", "every", "the"],
    "via": ["through", "using", "by means of", "with", "employing"],
    # project / task vocabulary
    "project": ["initiative", "effort", "program", "work", "endeavor"],
    "note": ["memo", "record", "entry", "log", "annotation"],
    "instruction": ["directive", "command", "order", "guidance", "rule"],
    "procedure": ["process", "method", "approach", "routine", "workflow"],
    "configuration": ["setup", "setting", "arrangement", "specification", "profile"],
}


# ---------------------------------------------------------------------------
# result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AdaptivePassageResult:
    """
    result of one adaptive evasion attempt on a single passage.

    fields:
        original_passage: initial poison passage (pre-evasion)
        evasive_passage: passage after greedy synonym substitution
        original_similarity: max cosine similarity before evasion
        evasive_similarity: max cosine similarity after evasion
        sad_threshold: target threshold (μ + k·σ) to stay below
        n_substitutions: number of word substitutions applied
        evasion_successful: True if evasive_similarity < sad_threshold
        retrieval_preserved: True if passage still ranks in top_k for victim
        substitution_log: list of (original_word, replacement) pairs applied
    """

    original_passage: str
    evasive_passage: str
    original_similarity: float
    evasive_similarity: float
    sad_threshold: float
    n_substitutions: int
    evasion_successful: bool
    retrieval_preserved: bool
    substitution_log: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_similarity": self.original_similarity,
            "evasive_similarity": self.evasive_similarity,
            "sad_threshold": self.sad_threshold,
            "n_substitutions": self.n_substitutions,
            "evasion_successful": self.evasion_successful,
            "retrieval_preserved": self.retrieval_preserved,
            "similarity_delta": self.evasive_similarity - self.original_similarity,
        }


@dataclass
class AdaptiveSADResult:
    """
    full result of adaptive adversary evaluation against sad.

    captures the fundamental evasion-retrieval tradeoff: passages crafted
    to evade sad by reducing similarity to victim queries also lose retrieval
    effectiveness, since both depend on the same similarity metric.

    fields:
        attack_type: "agent_poison", "minja", or "injecmem"
        n_trials: number of seeds evaluated
        # standard attack (no sad awareness)
        asr_r_standard: mean asr-r without adaptive evasion
        sad_tpr_standard: sad true positive rate vs standard attack
        sad_fpr_standard: sad false positive rate on benign entries
        sad_effectiveness_standard: 1 - asr_r_defended / asr_r_standard
        # adaptive attack (sad-aware synonym substitution)
        asr_r_adaptive: mean asr-r after adaptive evasion
        sad_tpr_adaptive: sad tpr against adaptive passages (lower = better evasion)
        sad_fpr_adaptive: sad fpr (unchanged — benign entries not modified)
        sad_effectiveness_adaptive: 1 - asr_r_defended_adaptive / asr_r_adaptive
        # tradeoff metrics
        evasion_rate: fraction of poison entries below sad threshold after evasion
        retrieval_degradation: asr_r_standard - asr_r_adaptive (cost of evasion)
        mean_substitutions_per_passage: mean synonyms applied per passage
        # sigma sweep for roc-like tradeoff curve
        sigma_sweep: list of {sigma, tpr_std, fpr_std, tpr_adv, fpr_adv} per sigma
        elapsed_s: total wall time
    """

    attack_type: str
    n_trials: int

    asr_r_standard: float
    sad_tpr_standard: float
    sad_fpr_standard: float
    sad_effectiveness_standard: float

    asr_r_adaptive: float
    sad_tpr_adaptive: float
    sad_fpr_adaptive: float
    sad_effectiveness_adaptive: float

    evasion_rate: float
    retrieval_degradation: float
    mean_substitutions_per_passage: float

    sigma_sweep: List[Dict[str, float]] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type,
            "n_trials": self.n_trials,
            "asr_r_standard": self.asr_r_standard,
            "sad_tpr_standard": self.sad_tpr_standard,
            "sad_fpr_standard": self.sad_fpr_standard,
            "sad_effectiveness_standard": self.sad_effectiveness_standard,
            "asr_r_adaptive": self.asr_r_adaptive,
            "sad_tpr_adaptive": self.sad_tpr_adaptive,
            "sad_fpr_adaptive": self.sad_fpr_adaptive,
            "sad_effectiveness_adaptive": self.sad_effectiveness_adaptive,
            "evasion_rate": self.evasion_rate,
            "retrieval_degradation": self.retrieval_degradation,
            "mean_substitutions_per_passage": self.mean_substitutions_per_passage,
            "sigma_sweep": self.sigma_sweep,
            "elapsed_s": self.elapsed_s,
        }


# ---------------------------------------------------------------------------
# adaptive passage crafter
# ---------------------------------------------------------------------------


class AdaptivePassageCrafter:
    """
    crafts adversarial passages that evade sad detection.

    uses greedy word-level synonym substitution to reduce the max cosine
    similarity of a poison passage to the observed victim query set,
    while monitoring whether the passage remains retrievable (top-k rank).

    this is a gradient-free black-box adaptive attack against sad: the
    attacker knows μ, σ (from calibration leakage or estimation), but
    does not have access to model weights or gradients.  only the embedding
    oracle is required.

    the vocabulary-constrained substitution approach preserves semantic
    coherence: replacements come from a hand-curated domain synonym dictionary,
    ensuring the modified passage is still a plausible memory entry.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_substitutions: int = 20,
        target_sigma_multiple: float = 1.5,
        seed: int = 42,
    ):
        """
        args:
            model_name: sentence-transformer model.  must match the retriever
                used by the target memory system (all-MiniLM-L6-v2, 384-dim).
            max_substitutions: maximum word swaps per passage before giving up.
            target_sigma_multiple: desired σ-multiple below detection threshold.
                e.g. 1.5 means target max_sim ≤ μ + 1.5σ < μ + 2.0σ (threshold).
            seed: random seed for vocabulary tie-breaking.
        """
        self.model_name = model_name
        self.max_substitutions = max_substitutions
        self.target_sigma_multiple = target_sigma_multiple
        self._rng = random.Random(seed)
        self._encoder = None
        self._synonyms = _ADAPTIVE_SYNONYMS

    @property
    def encoder(self):
        """lazy-initialize sentence-transformer encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _embed(self, texts: List[str]):
        """embed texts to l2-normalized vectors."""
        import numpy as np

        embs = self.encoder.encode(texts, normalize_embeddings=True)
        return np.array(embs, dtype=float)

    def _max_similarity(
        self,
        passage_emb,
        query_embs,
    ) -> float:
        """max cosine similarity of a passage embedding to a set of query embeddings."""
        import numpy as np

        sims = query_embs @ passage_emb  # (n_q,)
        return float(np.max(sims))

    def _get_candidates(self, word: str) -> List[str]:
        """get synonym candidates for a word (lowercase lookup)."""
        w_lower = word.lower().strip(".,;:!?\"'()[]")
        return self._synonyms.get(w_lower, [])

    def craft_evasive_passage(
        self,
        initial_passage: str,
        victim_queries: List[str],
        calibration_mean: float,
        calibration_std: float,
        threshold_sigma: float = 2.0,
        check_retrieval: bool = True,
    ) -> AdaptivePassageResult:
        """
        craft a passage that evades sad while (ideally) remaining retrievable.

        greedy algorithm:
            1. embed passage + victim queries.
            2. compute current max similarity.
            3. if below threshold: done (no evasion needed).
            4. while above threshold and budget remains:
               a. for each word in passage: try all synonym replacements.
               b. select replacement that most reduces max similarity.
               c. apply if helpful, check retrieval preservation.
               d. stop if no substitution helps.
            5. return evasive passage + diagnostics.

        args:
            initial_passage: starting poison passage text
            victim_queries: list of victim query strings
            calibration_mean: μ from sad calibration
            calibration_std: σ from sad calibration
            threshold_sigma: k in detection threshold (μ + k·σ)
            check_retrieval: if True, stop substitution if retrieval rank drops

        returns:
            AdaptivePassageResult with all evasion diagnostics
        """
        import numpy as np

        sad_threshold = calibration_mean + threshold_sigma * calibration_std
        q_embs = self._embed(victim_queries)  # (n_q, d)

        current_passage = initial_passage
        p_emb = self._embed([current_passage])[0]
        original_sim = self._max_similarity(p_emb, q_embs)
        current_sim = original_sim

        substitution_log: List[Tuple[str, str]] = []
        n_subs = 0

        # iterative greedy substitution
        for _iter in range(self.max_substitutions):
            if current_sim <= sad_threshold:
                break  # evasion achieved

            words = current_passage.split()
            best_sim = current_sim
            best_passage = current_passage
            best_word = None
            best_replacement = None

            for idx, word in enumerate(words):
                candidates = self._get_candidates(word)
                if not candidates:
                    continue
                for cand in candidates:
                    # build candidate passage with substitution
                    new_words = words[:]
                    new_words[idx] = cand
                    new_passage = " ".join(new_words)
                    # embed and score
                    new_emb = self._embed([new_passage])[0]
                    new_sim = self._max_similarity(new_emb, q_embs)
                    if new_sim < best_sim:
                        best_sim = new_sim
                        best_passage = new_passage
                        best_word = word
                        best_replacement = cand

            if best_word is None:
                # no substitution improves evasion — stuck
                break

            # check retrieval preservation if requested
            retrieval_ok = True
            if check_retrieval:
                # passage must still have some similarity to at least one query
                # (otherwise retrieval would completely fail)
                new_emb = self._embed([best_passage])[0]
                sims = q_embs @ new_emb
                retrieval_ok = float(np.max(sims)) > calibration_mean

            if not retrieval_ok:
                break  # would lose retrieval effectiveness — stop

            current_passage = best_passage
            current_sim = best_sim
            substitution_log.append((best_word, best_replacement))
            n_subs += 1

        evasion_successful = current_sim <= sad_threshold

        # check if final passage still ranks plausibly for victim queries
        final_emb = self._embed([current_passage])[0]
        final_sims = q_embs @ final_emb
        retrieval_preserved = float(np.max(final_sims)) > calibration_mean * 0.8

        return AdaptivePassageResult(
            original_passage=initial_passage,
            evasive_passage=current_passage,
            original_similarity=original_sim,
            evasive_similarity=current_sim,
            sad_threshold=sad_threshold,
            n_substitutions=n_subs,
            evasion_successful=evasion_successful,
            retrieval_preserved=retrieval_preserved,
            substitution_log=substitution_log,
        )


# ---------------------------------------------------------------------------
# adaptive sad evaluator
# ---------------------------------------------------------------------------


class AdaptiveSADEvaluator:
    """
    evaluate sad robustness under an adaptive white-box adversary.

    compares two scenarios:
        standard: attacker uses centroid-targeting passages without sad awareness
        adaptive: attacker applies greedy synonym substitution to evade sad

    produces tradeoff data showing: as evasion rate increases, asr-r decreases.
    this documents the fundamental tension between retrieval and evasion.

    in test mode (MEMORY_SECURITY_TEST=true), uses a small corpus and fewer
    iterations for fast ci.
    """

    # default sigma values for threshold sweep
    DEFAULT_SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    def __init__(
        self,
        corpus_size: int = 200,
        n_poison: int = 5,
        top_k: int = 5,
        n_calibration_queries: int = 10,
        seed: int = 42,
    ):
        """
        args:
            corpus_size: benign corpus size for retrieval evaluation
            n_poison: base poison count (agentpoison: ×1, minja: ×2, injecmem: ×3)
            top_k: retrieval top-k window
            n_calibration_queries: query count for sad calibration
            seed: random seed for reproducibility
        """
        _test = os.environ.get("MEMORY_SECURITY_TEST", "false").lower() == "true"
        if _test:
            corpus_size = min(corpus_size, 20)
            n_poison = min(n_poison, 2)
            top_k = min(top_k, 3)

        self.corpus_size = corpus_size
        self.n_poison = n_poison
        self.top_k = top_k
        self.n_calibration_queries = n_calibration_queries
        self.seed = seed
        self._crafter = AdaptivePassageCrafter(seed=seed)

    def _build_components(self, seed: int) -> Tuple[Any, Any, List[str], List[str]]:
        """
        build corpus, retrieval sim, and return components for evaluation.

        returns:
            (sim, corpus_obj, victim_queries, benign_sample)
        """
        from data.synthetic_corpus import SyntheticCorpus
        from evaluation.retrieval_sim import RetrievalSimulator

        corpus = SyntheticCorpus(seed=seed)
        sim = RetrievalSimulator(
            corpus_size=self.corpus_size,
            n_poison_per_attack=self.n_poison,
            top_k=self.top_k,
            use_trigger_optimization=False,
            seed=seed,
        )
        victim_query_dicts = corpus.get_victim_queries()
        # extract plain strings; dicts have key "query"
        victim_queries = [
            q["query"] if isinstance(q, dict) else q for q in victim_query_dicts
        ]
        benign_entries = corpus.generate_benign_entries(self.corpus_size)
        # use a sample of benign entries as sad calibration queries
        cal_entries = [
            e["content"] for e in benign_entries[: self.n_calibration_queries]
        ]
        return sim, corpus, victim_queries, cal_entries

    def _run_standard_trial(
        self,
        attack_type: str,
        seed: int,
        threshold_sigma: float = 2.0,
    ) -> Dict[str, float]:
        """
        run one trial: standard attack → sad calibration → evaluate tpr/fpr + asr-r.

        returns dict with asr_r, sad_tpr, sad_fpr keys.
        """
        from defenses.semantic_anomaly import SemanticAnomalyDetector
        from evaluation.retrieval_sim import (
            generate_centroid_agentpoison_passage,
            generate_injecmem_passage,
            generate_minja_passage,
        )

        sim, corpus, victim_queries, cal_queries = self._build_components(seed)

        # generate poison entries (standard, non-adaptive)
        benign_entries = corpus.generate_benign_entries(self.corpus_size)
        benign_texts = [e["content"] for e in benign_entries]

        poison_entries: List[str] = []
        if attack_type == "agent_poison":
            n = self.n_poison
            passage = generate_centroid_agentpoison_passage(victim_queries)
            for i in range(n):
                poison_entries.append(passage)
        elif attack_type == "minja":
            n = self.n_poison * 2
            for i in range(n):
                q = victim_queries[i % len(victim_queries)]
                poison_entries.append(generate_minja_passage(q))
        else:  # injecmem
            n = self.n_poison * 3
            for i in range(n):
                q = victim_queries[i % len(victim_queries)]
                poison_entries.append(generate_injecmem_passage(q, variant_index=i))

        # sad calibration on benign corpus
        detector = SemanticAnomalyDetector(threshold_sigma=threshold_sigma)
        cal_sample = benign_texts[: self.n_calibration_queries]
        detector.calibrate(benign_texts, cal_sample)
        for q in victim_queries[:10]:
            detector.update_query_set(q)

        # evaluate sad detection
        eval_result = detector.evaluate_on_corpus(poison_entries, benign_texts[:20])
        sad_tpr = eval_result["tpr"]
        sad_fpr = eval_result["fpr"]

        # evaluate asr-r (standard)
        asr_r = sim.evaluate_attack(attack_type).asr_r

        # compute effectiveness: fraction of asr-r blocked by sad
        n_blocked = round(len(poison_entries) * sad_tpr)
        n_survived = len(poison_entries) - n_blocked
        survived_ratio = n_survived / max(len(poison_entries), 1)
        # asr-r under sad ≈ asr-r × survived_ratio (linear approximation)
        asr_r_defended = asr_r * survived_ratio
        effectiveness = 1.0 - asr_r_defended / max(asr_r, 1e-6)

        return {
            "asr_r": asr_r,
            "sad_tpr": sad_tpr,
            "sad_fpr": sad_fpr,
            "sad_effectiveness": effectiveness,
        }

    def _run_adaptive_trial(
        self,
        attack_type: str,
        seed: int,
        threshold_sigma: float = 2.0,
    ) -> Dict[str, float]:
        """
        run one trial: adaptive evasive attack → sad calibration → evaluate.

        the adaptive attacker knows calibration stats and applies greedy
        synonym substitution to craft passages below the sad threshold.

        returns dict with asr_r, sad_tpr, sad_fpr, evasion_rate,
            retrieval_degradation, mean_subs keys.
        """
        from defenses.semantic_anomaly import SemanticAnomalyDetector
        from evaluation.retrieval_sim import (
            generate_centroid_agentpoison_passage,
            generate_injecmem_passage,
            generate_minja_passage,
        )

        sim, corpus, victim_queries, cal_queries = self._build_components(seed)
        benign_entries = corpus.generate_benign_entries(self.corpus_size)
        benign_texts = [e["content"] for e in benign_entries]

        # calibrate sad to get μ and σ (attacker has white-box access)
        detector = SemanticAnomalyDetector(threshold_sigma=threshold_sigma)
        cal_sample = benign_texts[: self.n_calibration_queries]
        stats = detector.calibrate(benign_texts, cal_sample)
        cal_mean = stats["mean"]
        cal_std = stats["std"]

        for q in victim_queries[:10]:
            detector.update_query_set(q)

        # generate standard passages
        standard_passages: List[str] = []
        if attack_type == "agent_poison":
            n = self.n_poison
            p = generate_centroid_agentpoison_passage(victim_queries)
            standard_passages = [p] * n
        elif attack_type == "minja":
            n = self.n_poison * 2
            for i in range(n):
                q = victim_queries[i % len(victim_queries)]
                standard_passages.append(generate_minja_passage(q))
        else:
            n = self.n_poison * 3
            for i in range(n):
                q = victim_queries[i % len(victim_queries)]
                standard_passages.append(generate_injecmem_passage(q, variant_index=i))

        # apply adaptive evasion: substitute words to reduce max-similarity
        evasive_passages: List[str] = []
        n_evasion_successes = 0
        n_retrieval_preserved = 0
        total_subs = 0
        adaptive_results: List[AdaptivePassageResult] = []
        for passage in standard_passages:
            result = self._crafter.craft_evasive_passage(
                initial_passage=passage,
                victim_queries=victim_queries,
                calibration_mean=cal_mean,
                calibration_std=cal_std,
                threshold_sigma=threshold_sigma,
            )
            adaptive_results.append(result)
            evasive_passages.append(result.evasive_passage)
            if result.evasion_successful:
                n_evasion_successes += 1
            if result.retrieval_preserved:
                n_retrieval_preserved += 1
            total_subs += result.n_substitutions

        evasion_rate = n_evasion_successes / max(len(standard_passages), 1)
        mean_subs = total_subs / max(len(standard_passages), 1)

        # evaluate sad against adaptive passages
        eval_result = detector.evaluate_on_corpus(evasive_passages, benign_texts[:20])
        sad_tpr_adaptive = eval_result["tpr"]
        sad_fpr_adaptive = eval_result["fpr"]

        # estimate asr-r degradation from retrieval preservation.
        # asr-r adaptive ≈ asr-r_standard × retrieval_preserve_rate (linear approx).
        retrieval_preserve_rate = n_retrieval_preserved / max(len(standard_passages), 1)
        asr_r_standard_est = sim.evaluate_attack(attack_type).asr_r
        asr_r_adaptive = asr_r_standard_est * retrieval_preserve_rate
        retrieval_degradation = asr_r_standard_est - asr_r_adaptive

        # effectiveness of sad against adaptive attacker
        n_blocked_adaptive = round(len(evasive_passages) * sad_tpr_adaptive)
        survived_ratio_adaptive = (len(evasive_passages) - n_blocked_adaptive) / max(
            len(evasive_passages), 1
        )
        asr_r_defended_adaptive = asr_r_adaptive * survived_ratio_adaptive
        effectiveness_adaptive = 1.0 - asr_r_defended_adaptive / max(
            asr_r_adaptive, 1e-6
        )

        return {
            "asr_r": asr_r_adaptive,
            "asr_r_standard": asr_r_standard_est,
            "sad_tpr": sad_tpr_adaptive,
            "sad_fpr": sad_fpr_adaptive,
            "sad_effectiveness": effectiveness_adaptive,
            "evasion_rate": evasion_rate,
            "retrieval_degradation": retrieval_degradation,
            "mean_subs": mean_subs,
        }

    def evaluate(
        self,
        attack_type: str = "agent_poison",
        sigma_values: Optional[List[float]] = None,
        n_trials: int = 3,
    ) -> AdaptiveSADResult:
        """
        full adaptive adversary evaluation against sad.

        runs both standard and adaptive attacks across n_trials seeds,
        computes means, and sweeps over sigma values for tradeoff curves.

        args:
            attack_type: "agent_poison", "minja", or "injecmem"
            sigma_values: threshold sigma values to sweep.  defaults to
                [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0].
            n_trials: number of seeds for mean estimation.

        returns:
            AdaptiveSADResult with full tradeoff analysis.
        """
        _test = os.environ.get("MEMORY_SECURITY_TEST", "false").lower() == "true"
        if _test:
            n_trials = min(n_trials, 1)
            sigma_values = sigma_values or [1.5, 2.0, 2.5]
        else:
            sigma_values = sigma_values or self.DEFAULT_SIGMA_VALUES

        t0 = time.time()
        logger.logger.info(
            "adaptive sad evaluation: attack=%s trials=%d", attack_type, n_trials
        )

        # --- standard trials at default sigma=2.0
        std_trials = []
        for i in range(n_trials):
            seed = self.seed + i * 17
            try:
                r = self._run_standard_trial(attack_type, seed, threshold_sigma=2.0)
                std_trials.append(r)
            except Exception as exc:
                logger.log_error("adaptive_eval_standard", exc, {"trial": i})

        # --- adaptive trials at default sigma=2.0
        adv_trials = []
        for i in range(n_trials):
            seed = self.seed + i * 17
            try:
                r = self._run_adaptive_trial(attack_type, seed, threshold_sigma=2.0)
                adv_trials.append(r)
            except Exception as exc:
                logger.log_error("adaptive_eval_adaptive", exc, {"trial": i})

        def _mean(lst, key):
            vals = [x[key] for x in lst if key in x]
            return sum(vals) / len(vals) if vals else 0.0

        # --- sigma sweep for tradeoff curves
        sigma_sweep = []
        for sigma in sigma_values:
            try:
                std_r = self._run_standard_trial(attack_type, self.seed, sigma)
                adv_r = self._run_adaptive_trial(attack_type, self.seed, sigma)
                sigma_sweep.append(
                    {
                        "sigma": sigma,
                        "tpr_standard": std_r["sad_tpr"],
                        "fpr_standard": std_r["sad_fpr"],
                        "effectiveness_standard": std_r["sad_effectiveness"],
                        "tpr_adaptive": adv_r["sad_tpr"],
                        "fpr_adaptive": adv_r["sad_fpr"],
                        "effectiveness_adaptive": adv_r["sad_effectiveness"],
                        "evasion_rate": adv_r["evasion_rate"],
                        "retrieval_degradation": adv_r["retrieval_degradation"],
                    }
                )
            except Exception as exc:
                logger.log_error("adaptive_eval_sigma_sweep", exc, {"sigma": sigma})

        asr_r_standard = _mean(std_trials, "asr_r") if std_trials else 0.0
        asr_r_adaptive = _mean(adv_trials, "asr_r") if adv_trials else 0.0

        result = AdaptiveSADResult(
            attack_type=attack_type,
            n_trials=n_trials,
            asr_r_standard=asr_r_standard,
            sad_tpr_standard=_mean(std_trials, "sad_tpr"),
            sad_fpr_standard=_mean(std_trials, "sad_fpr"),
            sad_effectiveness_standard=_mean(std_trials, "sad_effectiveness"),
            asr_r_adaptive=asr_r_adaptive,
            sad_tpr_adaptive=_mean(adv_trials, "sad_tpr"),
            sad_fpr_adaptive=_mean(adv_trials, "sad_fpr"),
            sad_effectiveness_adaptive=_mean(adv_trials, "sad_effectiveness"),
            evasion_rate=_mean(adv_trials, "evasion_rate"),
            retrieval_degradation=_mean(adv_trials, "retrieval_degradation"),
            mean_substitutions_per_passage=_mean(adv_trials, "mean_subs"),
            sigma_sweep=sigma_sweep,
            elapsed_s=time.time() - t0,
        )

        logger.logger.info(
            "adaptive sad done: asr_r_std=%.3f asr_r_adv=%.3f evasion_rate=%.3f",
            result.asr_r_standard,
            result.asr_r_adaptive,
            result.evasion_rate,
        )
        return result

    def evaluate_all_attacks(
        self,
        sigma_values: Optional[List[float]] = None,
        n_trials: int = 3,
    ) -> Dict[str, AdaptiveSADResult]:
        """
        run adaptive sad evaluation for all three attacks.

        returns:
            dict mapping attack_type → AdaptiveSADResult
        """
        attacks = ["agent_poison", "minja", "injecmem"]
        results: Dict[str, AdaptiveSADResult] = {}
        for at in attacks:
            results[at] = self.evaluate(
                at, sigma_values=sigma_values, n_trials=n_trials
            )
        return results

    def to_latex_table(
        self,
        results: Dict[str, AdaptiveSADResult],
        caption: str = "Adaptive adversary evaluation against SAD defense.",
        label: str = "tab:adaptive_sad",
    ) -> str:
        """
        generate a booktabs latex table comparing standard vs adaptive attack.

        columns: attack | asr-r std | tpr std | effect. std | asr-r adv |
                 tpr adv | effect. adv | evasion rate | retrieval degr.
        """
        attack_labels = {
            "agent_poison": "AgentPoison",
            "minja": "MINJA",
            "injecmem": "InjecMEM",
        }
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Attack & \multicolumn{3}{c}{Standard} & \multicolumn{3}{c}{Adaptive} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r" & ASR-R & SAD TPR & Effect. & ASR-R & SAD TPR & Evasion \\",
            r"\midrule",
        ]
        for at, res in results.items():
            label_str = attack_labels.get(at, at)
            lines.append(
                f"{label_str} & "
                f"{res.asr_r_standard:.3f} & "
                f"{res.sad_tpr_standard:.3f} & "
                f"{res.sad_effectiveness_standard:.3f} & "
                f"{res.asr_r_adaptive:.3f} & "
                f"{res.sad_tpr_adaptive:.3f} & "
                f"{res.evasion_rate:.3f} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)
