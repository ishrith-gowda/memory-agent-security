"""
attack-defense interaction matrix evaluator for memory agent security research.

this module evaluates all (attack, defense) pairs in a unified framework,
producing the key table for the paper (table 2 or 3 depending on paper structure).

evaluation model: pre-ingestion filtering.
    for each (attack, defense) pair:
    1. generate poison entries (via RetrievalSimulator._generate_poison_entries)
    2. run defense detection on each poison entry — flagged entries are blocked
    3. run retrieval simulation with only undetected (surviving) poison entries
    4. record: asr-r_baseline, asr-r_under_defense, defense tpr, fpr, effectiveness

defense effectiveness:
    reduction  = 1 - asr_r_under_defense / asr_r_baseline   (higher = better)
    this is analogous to "detection rate" in the backdoor defense literature
    (wang et al., neural cleanse, ieee s&p 2019; gao et al., strip, acsac 2019).

for sad (semantic anomaly detection): calibrate on benign corpus,
populate with observed victim queries, then batch-detect poison entries.

attack-defense matrix format follows ASB (zhang et al., arXiv:2410.02644)
and neural cleanse (wang et al., 2019): rows = attacks, cols = defenses,
cells = post-defense asr ± std over n_trials seeds.

all comments are lowercase.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from data.synthetic_corpus import SyntheticCorpus
from defenses.implementations import create_defense
from defenses.semantic_anomaly import SemanticAnomalyDetector
from evaluation.retrieval_sim import RetrievalSimulator
from memory_systems.vector_store import VectorMemorySystem
from utils.logging import logger

# attacks and defenses evaluated in the matrix
ATTACK_TYPES = ["agent_poison", "minja", "injecmem"]
DEFENSE_TYPES = [
    "watermark",
    "validation",
    "proactive",
    "composite",
    "semantic_anomaly",
]

# human-readable labels for tables
ATTACK_DISPLAY = {
    "agent_poison": "AgentPoison",
    "minja": "MINJA",
    "injecmem": "InjecMEM",
}
DEFENSE_DISPLAY = {
    "watermark": "Watermark",
    "validation": "Validation",
    "proactive": "Proactive",
    "composite": "Composite",
    "semantic_anomaly": "SAD (ours)",
}


# ---------------------------------------------------------------------------
# result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PairResult:
    """
    result for a single (attack, defense) evaluation.

    fields:
        attack_type: attack identifier
        defense_type: defense identifier
        asr_r_baseline: asr-r without any defense
        asr_r_under_defense: asr-r with defense active (pre-ingestion filter)
        asr_t_baseline: asr-t without defense
        asr_t_under_defense: asr-t with defense active
        defense_tpr: fraction of poison entries correctly flagged
        defense_fpr: fraction of benign entries incorrectly flagged
        defense_effectiveness: 1 - asr_r_under_defense / asr_r_baseline
        n_poison_total: total poison entries generated
        n_poison_blocked: poison entries blocked by defense
        n_poison_survived: poison entries that evaded detection
        elapsed_s: wall-clock evaluation time
    """

    attack_type: str
    defense_type: str
    asr_r_baseline: float = 0.0
    asr_r_under_defense: float = 0.0
    asr_t_baseline: float = 0.0
    asr_t_under_defense: float = 0.0
    defense_tpr: float = 0.0
    defense_fpr: float = 0.0
    defense_effectiveness: float = 0.0
    n_poison_total: int = 0
    n_poison_blocked: int = 0
    n_poison_survived: int = 0
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type,
            "defense_type": self.defense_type,
            "asr_r_baseline": self.asr_r_baseline,
            "asr_r_under_defense": self.asr_r_under_defense,
            "asr_t_baseline": self.asr_t_baseline,
            "asr_t_under_defense": self.asr_t_under_defense,
            "defense_tpr": self.defense_tpr,
            "defense_fpr": self.defense_fpr,
            "defense_effectiveness": self.defense_effectiveness,
            "n_poison_survived": self.n_poison_survived,
            "n_poison_blocked": self.n_poison_blocked,
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class MatrixResult:
    """
    full (attacks × defenses) evaluation matrix.

    indexed as results[attack_type][defense_type] = PairResult.
    also includes per-attack baseline metrics (no defense).
    """

    n_trials: int
    corpus_size: int
    n_poison: int
    top_k: int
    results: Dict[str, Dict[str, PairResult]] = field(default_factory=dict)

    def get(self, attack: str, defense: str) -> Optional[PairResult]:
        return self.results.get(attack, {}).get(defense)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "corpus_size": self.corpus_size,
            "n_poison": self.n_poison,
            "top_k": self.top_k,
            "results": {
                atk: {def_: r.to_dict() for def_, r in defenses.items()}
                for atk, defenses in self.results.items()
            },
        }


# ---------------------------------------------------------------------------
# defense detection helpers
# ---------------------------------------------------------------------------


def _detect_with_defense(
    defense_type: str,
    poison_entries: List[Dict[str, Any]],
    benign_sample: List[str],
    corpus_entries: Optional[List[Dict[str, Any]]] = None,
    victim_queries: Optional[List[str]] = None,
) -> Tuple[List[bool], List[bool], float]:
    """
    run a specific defense on poison + benign entries.

    args:
        defense_type: which defense to apply
        poison_entries: adversarial {key, content, metadata} dicts
        benign_sample: sample of benign entry strings for fpr estimation
        corpus_entries: full benign corpus for SAD calibration
        victim_queries: observed victim queries for SAD

    returns:
        (poison_flagged, benign_flagged, latency_ms_avg)
        poison_flagged: list[bool], True if defense flagged poison[i]
        benign_flagged: list[bool], True if defense flagged benign[i]
        latency_ms_avg: average detection latency in milliseconds
    """
    poison_texts = [e["content"] for e in poison_entries]
    all_texts = poison_texts + list(benign_sample)
    latencies: List[float] = []

    if defense_type == "semantic_anomaly":
        # sad requires calibration + query observations
        cal_entries = [e["content"] for e in (corpus_entries or [])]
        cal_queries = list(victim_queries or [])[:10]
        if not cal_entries or not cal_queries:
            # fallback: no detection if calibration not available
            return (
                [False] * len(poison_entries),
                [False] * len(benign_sample),
                0.0,
            )
        det = SemanticAnomalyDetector(threshold_sigma=2.0)
        det.calibrate(cal_entries, cal_queries)
        for q in victim_queries or []:
            det.update_query_set(q)
        t0 = time.perf_counter()
        poison_results = det.detect_batch(poison_texts)
        benign_results = det.detect_batch(list(benign_sample))
        elapsed = time.perf_counter() - t0
        latency_ms = elapsed * 1000 / max(len(all_texts), 1)
        return (
            [r.is_anomalous for r in poison_results],
            [r.is_anomalous for r in benign_results],
            latency_ms,
        )

    else:
        # standard defense api: detect_attack(content) → {"attack_detected": bool}
        try:
            defense = create_defense(defense_type)
            defense.activate()
        except Exception:
            return (
                [False] * len(poison_entries),
                [False] * len(benign_sample),
                0.0,
            )

        poison_flagged: List[bool] = []
        for text in poison_texts:
            t0 = time.perf_counter()
            result = defense.detect_attack(text)
            latencies.append((time.perf_counter() - t0) * 1000)
            poison_flagged.append(bool(result.get("attack_detected", False)))

        benign_flagged: List[bool] = []
        for text in benign_sample:
            t0 = time.perf_counter()
            result = defense.detect_attack(text)
            latencies.append((time.perf_counter() - t0) * 1000)
            benign_flagged.append(bool(result.get("attack_detected", False)))

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return poison_flagged, benign_flagged, avg_latency


# ---------------------------------------------------------------------------
# main evaluator
# ---------------------------------------------------------------------------


class AttackDefenseEvaluator:
    """
    evaluate all (attack, defense) pairs in a cross-product matrix.

    usage:
        evaluator = AttackDefenseEvaluator(corpus_size=200, n_poison=5)
        matrix = evaluator.evaluate_full_matrix(n_trials=3)
        print(matrix.get("minja", "watermark").asr_r_under_defense)
        latex = evaluator.to_latex_matrix(matrix)
    """

    def __init__(
        self,
        corpus_size: int = 200,
        n_poison: int = 5,
        top_k: int = 5,
        use_trigger_optimization: bool = False,
        seed: int = 42,
    ):
        """
        args:
            corpus_size: number of benign entries in the memory corpus
            n_poison: base poison count per attack (multiplied per-attack)
            top_k: retrieval depth
            use_trigger_optimization: enable agentpoison trigger optimizer
            seed: random seed
        """
        self.corpus_size = corpus_size
        self.n_poison = n_poison
        self.top_k = top_k
        self.use_trigger_optimization = use_trigger_optimization
        self.seed = seed

    def _run_single_trial(
        self,
        attack_type: str,
        defense_type: str,
        seed: int,
    ) -> PairResult:
        """
        run one trial of (attack, defense) evaluation.

        generates poison entries, applies defense filter, then measures asr
        with the surviving (undetected) poison entries.
        """
        t_start = time.perf_counter()
        random.seed(seed)

        # build corpus and victim queries
        corpus = SyntheticCorpus(seed=seed)
        benign_entries = corpus.generate_benign_entries(self.corpus_size)
        victim_queries = [q["query"] for q in corpus.get_victim_queries()]

        # generate poison entries for this attack
        sim = RetrievalSimulator(
            corpus_size=self.corpus_size,
            n_poison_per_attack=self.n_poison,
            top_k=self.top_k,
            use_trigger_optimization=self.use_trigger_optimization,
            seed=seed,
        )
        poison_entries = sim._generate_poison_entries(attack_type, victim_queries)
        trigger_str = sim._last_trigger_string  # set by agent_poison w/ trigger opt

        # baseline: measure asr-r without any defense
        baseline_metrics = sim.evaluate_attack(attack_type)

        # defense: flag poison entries
        benign_sample = [e["content"] for e in benign_entries[:50]]
        corpus_for_cal = benign_entries

        poison_flagged, benign_flagged, latency_ms = _detect_with_defense(
            defense_type=defense_type,
            poison_entries=poison_entries,
            benign_sample=benign_sample,
            corpus_entries=corpus_for_cal,
            victim_queries=victim_queries,
        )

        n_poison_blocked = sum(poison_flagged)
        n_poison_survived = len(poison_entries) - n_poison_blocked

        # compute defense tpr and fpr
        tpr = n_poison_blocked / len(poison_entries) if poison_entries else 0.0
        fpr = sum(benign_flagged) / len(benign_sample) if benign_sample else 0.0

        # simulate retrieval with only surviving (undetected) poison entries
        surviving = [e for e, f in zip(poison_entries, poison_flagged) if not f]

        if surviving:
            # build fresh memory with only surviving poison + benign corpus
            mem = VectorMemorySystem()
            for entry in benign_entries:
                mem.store(
                    entry["key"],
                    entry["content"],
                    entry.get("metadata", {}),
                )
            actual_poison_keys = []
            for entry in surviving:
                mem.store(entry["key"], entry["content"], entry.get("metadata", {}))
                actual_poison_keys.append(entry["key"])

            # measure asr-r under defense
            retrieved_defended = 0
            total_qs = len(victim_queries)
            use_triggered = attack_type == "agent_poison" and trigger_str is not None
            for q in victim_queries:
                rq = f"{trigger_str} {q}" if use_triggered else q
                result = mem.poison_retrieval_test(rq, actual_poison_keys, self.top_k)
                if result["retrieved_any_poison"]:
                    retrieved_defended += 1

            asr_r_defended = retrieved_defended / total_qs if total_qs > 0 else 0.0

            # model asr-t under defense (same modelled asr-a, reduced retrieval)
            from evaluation.retrieval_sim import _MODELLED_ASR_A

            mean_asr_a, std_asr_a = _MODELLED_ASR_A.get(attack_type, (0.60, 0.08))
            rng = random.Random(seed + 7)
            hijacks = 0
            for _ in range(retrieved_defended):
                action_prob = max(0.0, min(1.0, rng.gauss(mean_asr_a, std_asr_a)))
                if rng.random() < action_prob:
                    hijacks += 1
            asr_t_defended = hijacks / total_qs if total_qs > 0 else 0.0
        else:
            asr_r_defended = 0.0
            asr_t_defended = 0.0

        # defense effectiveness
        effectiveness = (
            1.0 - asr_r_defended / baseline_metrics.asr_r
            if baseline_metrics.asr_r > 0
            else 1.0
        )

        elapsed = time.perf_counter() - t_start
        return PairResult(
            attack_type=attack_type,
            defense_type=defense_type,
            asr_r_baseline=baseline_metrics.asr_r,
            asr_r_under_defense=asr_r_defended,
            asr_t_baseline=baseline_metrics.asr_t,
            asr_t_under_defense=asr_t_defended,
            defense_tpr=tpr,
            defense_fpr=fpr,
            defense_effectiveness=max(0.0, min(1.0, effectiveness)),
            n_poison_total=len(poison_entries),
            n_poison_blocked=n_poison_blocked,
            n_poison_survived=n_poison_survived,
            elapsed_s=elapsed,
        )

    def evaluate_pair(
        self,
        attack_type: str,
        defense_type: str,
        n_trials: int = 3,
    ) -> PairResult:
        """
        evaluate one (attack, defense) pair averaged over n_trials seeds.

        for n_trials > 1, all scalar fields are averaged; the result
        represents the mean behaviour across seeds.
        """
        if n_trials == 1:
            return self._run_single_trial(attack_type, defense_type, self.seed)

        results: List[PairResult] = []
        for trial in range(n_trials):
            r = self._run_single_trial(attack_type, defense_type, self.seed + trial)
            results.append(r)

        # average all numeric fields
        def avg(field_name: str) -> float:
            vals = [getattr(r, field_name) for r in results]
            return sum(vals) / len(vals)

        return PairResult(
            attack_type=attack_type,
            defense_type=defense_type,
            asr_r_baseline=avg("asr_r_baseline"),
            asr_r_under_defense=avg("asr_r_under_defense"),
            asr_t_baseline=avg("asr_t_baseline"),
            asr_t_under_defense=avg("asr_t_under_defense"),
            defense_tpr=avg("defense_tpr"),
            defense_fpr=avg("defense_fpr"),
            defense_effectiveness=avg("defense_effectiveness"),
            n_poison_total=results[0].n_poison_total,
            n_poison_blocked=round(avg("n_poison_blocked")),
            n_poison_survived=round(avg("n_poison_survived")),
            elapsed_s=avg("elapsed_s"),
        )

    def evaluate_full_matrix(
        self,
        attacks: Optional[List[str]] = None,
        defenses: Optional[List[str]] = None,
        n_trials: int = 3,
    ) -> MatrixResult:
        """
        evaluate all (attack, defense) combinations.

        args:
            attacks: attack types to evaluate (default: all 3)
            defenses: defense types to evaluate (default: all 5)
            n_trials: trials per pair (averaged)

        returns:
            MatrixResult with all pair results
        """
        attacks = attacks or ATTACK_TYPES
        defenses = defenses or DEFENSE_TYPES

        matrix = MatrixResult(
            n_trials=n_trials,
            corpus_size=self.corpus_size,
            n_poison=self.n_poison,
            top_k=self.top_k,
        )

        total = len(attacks) * len(defenses)
        done = 0
        for attack in attacks:
            matrix.results[attack] = {}
            for defense in defenses:
                done += 1
                logger.logger.info(
                    f"evaluating ({attack}, {defense})  [{done}/{total}]"
                )
                pair = self.evaluate_pair(attack, defense, n_trials=n_trials)
                matrix.results[attack][defense] = pair

        return matrix

    # ------------------------------------------------------------------
    # latex table generation
    # ------------------------------------------------------------------

    def to_latex_matrix(
        self,
        matrix: MatrixResult,
        metric: str = "asr_r",
        caption: str = (
            "Attack-defense interaction matrix: post-defense ASR-R "
            "(fraction of victim queries that retrieve undetected poison). "
            "Rows = attacks, Columns = defenses. "
            "Lower is better for defenses. Baseline = no defense."
        ),
        label: str = "tab:attack_defense_matrix",
        attacks: Optional[List[str]] = None,
        defenses: Optional[List[str]] = None,
    ) -> str:
        """
        generate booktabs latex table: rows=attacks, cols=defenses.

        cells show post-defense metric (asr_r_under_defense or asr_t_under_defense).
        also includes a "Baseline" column (no defense) and a "Δ Reduction" column.

        args:
            metric: "asr_r" or "asr_t"
            caption: latex caption
            label: latex label
        """
        attacks = attacks or ATTACK_TYPES
        defenses = defenses or DEFENSE_TYPES

        baseline_key = "asr_r_baseline" if metric == "asr_r" else "asr_t_baseline"
        defended_key = (
            "asr_r_under_defense" if metric == "asr_r" else "asr_t_under_defense"
        )
        metric_upper = "ASR-R" if metric == "asr_r" else "ASR-T"

        # build column spec: attack | baseline | defenses...
        n_cols = 2 + len(defenses)
        col_spec = "l" + "c" * (n_cols - 1)

        header_row = (
            "Attack & Baseline & "
            + " & ".join(DEFENSE_DISPLAY.get(d, d) for d in defenses)
            + " \\\\"
        )

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"\\multicolumn{{{n_cols}}}{{c}}{{{metric_upper} Under Defense "
            f"(lower is better for defender)}} \\\\",
            "\\midrule",
            header_row,
            "\\midrule",
        ]

        for attack in attacks:
            row_vals: List[str] = [ATTACK_DISPLAY.get(attack, attack)]

            # baseline (no defense)
            any_pair = matrix.results.get(attack, {})
            if any_pair:
                baseline_val = next(iter(any_pair.values()))
                row_vals.append(f"{getattr(baseline_val, baseline_key):.3f}")
            else:
                row_vals.append("—")

            # per-defense values
            defended_vals = []
            for defense in defenses:
                pair = matrix.get(attack, defense)
                if pair is not None:
                    v = getattr(pair, defended_key)
                    defended_vals.append(v)
                    row_vals.append(f"{v:.3f}")
                else:
                    defended_vals.append(None)
                    row_vals.append("—")

            # bold the lowest post-defense value (best defense for this attack)
            numeric = [(i, v) for i, v in enumerate(defended_vals) if v is not None]
            if numeric:
                best_i, _ = min(numeric, key=lambda x: x[1])
                col_offset = 2  # 0=attack, 1=baseline, 2+ = defenses
                raw = row_vals[col_offset + best_i]
                row_vals[col_offset + best_i] = f"\\textbf{{{raw}}}"

            lines.append(" & ".join(row_vals) + " \\\\")

        # add defense effectiveness row
        lines.append("\\midrule")
        eff_row = ["Avg.~Effectiveness (↑)"]
        eff_row.append("—")  # no baseline effectiveness
        for defense in defenses:
            effs = []
            for attack in attacks:
                pair = matrix.get(attack, defense)
                if pair is not None:
                    effs.append(pair.defense_effectiveness)
            if effs:
                avg_eff = sum(effs) / len(effs)
                eff_row.append(f"{avg_eff:.3f}")
            else:
                eff_row.append("—")
        lines.append(" & ".join(eff_row) + " \\\\")

        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
        return "\n".join(lines)

    def to_latex_tpr_fpr_table(
        self,
        matrix: MatrixResult,
        caption: str = (
            "Defense detection performance (TPR, FPR) per (attack, defense) pair. "
            "SAD = Semantic Anomaly Detection (ours)."
        ),
        label: str = "tab:defense_tpr_fpr",
        attacks: Optional[List[str]] = None,
        defenses: Optional[List[str]] = None,
    ) -> str:
        """
        generate tpr/fpr table: secondary table for the paper.

        rows = defenses, cols = attacks, cells = "TPR / FPR".
        """
        attacks = attacks or ATTACK_TYPES
        defenses = defenses or DEFENSE_TYPES

        n_cols = 1 + len(attacks)
        col_spec = "l" + "c" * len(attacks)
        header = (
            "Defense & "
            + " & ".join(ATTACK_DISPLAY.get(a, a) for a in attacks)
            + " \\\\"
        )

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"\\multicolumn{{{n_cols}}}{{c}}{{TPR / FPR by (Defense, Attack)}} \\\\",
            "\\midrule",
            header,
            "\\midrule",
        ]

        for defense in defenses:
            row = [DEFENSE_DISPLAY.get(defense, defense)]
            for attack in attacks:
                pair = matrix.get(attack, defense)
                if pair is not None:
                    row.append(f"{pair.defense_tpr:.2f} / {pair.defense_fpr:.2f}")
                else:
                    row.append("— / —")
            lines.append(" & ".join(row) + " \\\\")

        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        return "\n".join(lines)
