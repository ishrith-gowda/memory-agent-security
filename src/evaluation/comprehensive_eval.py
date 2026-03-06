"""
comprehensive end-to-end evaluation for memory agent security.

this module is the master evaluation runner for the neurips / acm ccs paper.
it orchestrates all evaluation components into a single coherent pipeline:

    1. multi-trial attack evaluation (bootstrap 95% ci)
    2. full 3 × 5 attack-defense matrix
    3. watermark evasion analysis (three evasion strategies)
    4. adaptive adversary against sad (novel contribution)
    5. hyperparameter ablation studies
    6. latex table generation for all paper tables
    7. json serialization of all results

this module is designed to be run once per camera-ready deadline.  it is
deterministic (seed_base=42) and supports fast ci mode via MEMORY_SECURITY_TEST.

timing (approximate, without trigger optimization):
    quick mode (MEMORY_SECURITY_TEST=true): < 3 minutes
    full mode (n_seeds=5, n_trials=3 for matrix): 20-40 minutes depending on hardware

all comments are lowercase.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logging import logger

# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class ComprehensiveResult:
    """
    container for all evaluation results produced by ComprehensiveEvaluator.

    fields:
        attack_summaries: multi-trial bootstrap summaries per attack.
            keys: "agent_poison", "minja", "injecmem".
            values: serializable dicts with asr_r/asr_t/benign_acc ci fields.
        matrix_result_dict: serialized MatrixResult (attack × defense table).
        evasion_results: dict from evasion strategy name → EvasionResult summary.
        adaptive_sad_results: dict from attack_type → AdaptiveSADResult dict.
        ablation_results: dict from study_name → list of AblationPoint dicts.
        generated_at: iso timestamp of run completion.
        config: all hyperparameters and mode settings used for this run.
        elapsed_s: total wall time for the full evaluation.
    """

    attack_summaries: Dict[str, Any] = field(default_factory=dict)
    matrix_result_dict: Optional[Dict[str, Any]] = None
    evasion_results: Dict[str, Any] = field(default_factory=dict)
    adaptive_sad_results: Dict[str, Any] = field(default_factory=dict)
    ablation_results: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_summaries": self.attack_summaries,
            "matrix_result": self.matrix_result_dict,
            "evasion_results": self.evasion_results,
            "adaptive_sad_results": self.adaptive_sad_results,
            "ablation_results": self.ablation_results,
            "generated_at": self.generated_at,
            "config": self.config,
            "elapsed_s": self.elapsed_s,
        }

    def save_json(self, path: str) -> None:
        """save full result dict to json file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.logger.info("comprehensive results saved to %s", path)


# ---------------------------------------------------------------------------
# main evaluator
# ---------------------------------------------------------------------------


class ComprehensiveEvaluator:
    """
    master evaluation runner.

    runs all attacks, all defenses, evasion analysis, adaptive adversary
    evaluation, and hyperparameter ablations.  produces all result tables
    and figures needed for the paper submission.

    controlled by boolean flags so individual components can be skipped
    for faster iteration during paper writing.

    test mode (MEMORY_SECURITY_TEST=true):
        - reduces corpus_size, trial counts, and ablation grids
        - estimated runtime: < 3 minutes for all components
    """

    def __init__(
        self,
        corpus_size: int = 200,
        n_poison: int = 5,
        top_k: int = 5,
        n_seeds: int = 5,
        seed_base: int = 42,
        run_matrix: bool = True,
        run_evasion: bool = True,
        run_adaptive: bool = True,
        run_ablations: bool = True,
    ):
        """
        args:
            corpus_size: number of benign entries in synthetic corpus.
            n_poison: base poison count (scaled per attack: ×1/×2/×3).
            top_k: retrieval window size for vector search.
            n_seeds: seeds for multi-trial bootstrap evaluation.
            seed_base: base random seed (seeds: seed_base, seed_base+17, ...).
            run_matrix: if False, skip the 3×5 attack-defense matrix.
            run_evasion: if False, skip watermark evasion analysis.
            run_adaptive: if False, skip adaptive adversary against sad.
            run_ablations: if False, skip hyperparameter ablation studies.
        """
        _test = os.environ.get("MEMORY_SECURITY_TEST", "false").lower() == "true"
        if _test:
            corpus_size = min(corpus_size, 20)
            n_poison = min(n_poison, 2)
            top_k = min(top_k, 3)
            n_seeds = min(n_seeds, 2)

        self.corpus_size = corpus_size
        self.n_poison = n_poison
        self.top_k = top_k
        self.n_seeds = n_seeds
        self.seed_base = seed_base
        self.run_matrix = run_matrix
        self.run_evasion = run_evasion
        self.run_adaptive = run_adaptive
        self.run_ablations = run_ablations
        self._test = _test

    # ------------------------------------------------------------------
    # step 1: multi-trial attack evaluation
    # ------------------------------------------------------------------

    def _run_attack_evaluation(self) -> Dict[str, Any]:
        """
        run multi-trial bootstrap evaluation for all three attacks.

        returns dict mapping attack_type → serializable ci summary.
        """
        from evaluation.statistical import MultiTrialEvaluator

        logger.logger.info(
            "running multi-trial attack evaluation (n_seeds=%d)", self.n_seeds
        )
        mt_eval = MultiTrialEvaluator(
            corpus_size=self.corpus_size,
            n_poison=self.n_poison,
            top_k=self.top_k,
        )

        attack_summaries: Dict[str, Any] = {}
        for at in ["agent_poison", "minja", "injecmem"]:
            try:
                summary = mt_eval.evaluate_attack(
                    at,
                    n_trials=self.n_seeds,
                    seeds=[self.seed_base + i * 17 for i in range(self.n_seeds)],
                )
                # serialize BootstrapResult objects
                attack_summaries[at] = {
                    "asr_r": {
                        "mean": summary.asr_r.mean,
                        "lower": summary.asr_r.lower,
                        "upper": summary.asr_r.upper,
                    },
                    "asr_t": {
                        "mean": summary.asr_t.mean,
                        "lower": summary.asr_t.lower,
                        "upper": summary.asr_t.upper,
                    },
                    "benign_accuracy": {
                        "mean": summary.benign_accuracy.mean,
                        "lower": summary.benign_accuracy.lower,
                        "upper": summary.benign_accuracy.upper,
                    },
                    "injection_success_rate": summary.injection_success_rate,
                }
                logger.logger.info(
                    "  %s: asr_r=%.3f [%.3f, %.3f]",
                    at,
                    summary.asr_r.mean,
                    summary.asr_r.lower,
                    summary.asr_r.upper,
                )
            except Exception as exc:
                logger.log_error("multi_trial_eval", exc, {"attack": at})
                attack_summaries[at] = {"error": str(exc)}

        return attack_summaries

    # ------------------------------------------------------------------
    # step 2: attack-defense matrix
    # ------------------------------------------------------------------

    def _run_matrix(self) -> Optional[Dict[str, Any]]:
        """
        run full 3 × 5 attack-defense matrix evaluation.

        returns serialized MatrixResult dict.
        """
        from evaluation.attack_defense_matrix import AttackDefenseEvaluator

        n_trials = 2 if self._test else 3
        logger.logger.info(
            "running attack-defense matrix (n_trials=%d per pair)", n_trials
        )

        evaluator = AttackDefenseEvaluator(
            corpus_size=self.corpus_size,
            n_poison=self.n_poison,
            top_k=self.top_k,
            use_trigger_optimization=False,
            seed=self.seed_base,
        )

        attacks = ["agent_poison", "minja", "injecmem"]
        defenses = [
            "watermark",
            "validation",
            "proactive",
            "composite",
            "semantic_anomaly",
        ]

        try:
            matrix = evaluator.evaluate_full_matrix(
                attacks=attacks,
                defenses=defenses,
                n_trials=n_trials,
            )
            return matrix.to_dict()
        except Exception as exc:
            logger.log_error("matrix_eval", exc, {})
            return None

    # ------------------------------------------------------------------
    # step 3: watermark evasion
    # ------------------------------------------------------------------

    def _run_evasion(self) -> Dict[str, Any]:
        """
        run watermark evasion analysis (paraphrase + dilution + adaptive substitution).

        returns dict mapping evasion_type → EvasionResult summary.
        """
        from evaluation.evasion_eval import WatermarkEvasionEvaluator
        from watermark.watermarking import create_watermark_encoder

        logger.logger.info("running watermark evasion analysis")
        encoder = create_watermark_encoder("unigram")

        # reference content (realistic memory entry length)
        _ref = (
            "the memory agent maintains a comprehensive record of user preferences "
            "task history calendar events and interaction context across multiple "
            "sessions for personalised response generation and contextual retrieval. "
            "this entry covers scheduling preferences communication style guidelines "
            "and historical decision patterns as observed in prior agent interactions."
        )

        n_samples = 10 if self._test else 30
        evaluator = WatermarkEvasionEvaluator(
            encoder=encoder, n_samples=n_samples, seed=self.seed_base
        )

        # generate watermarked and clean samples
        wm_samples: List[str] = []
        clean_samples: List[str] = []
        for i in range(n_samples):
            wm_id = f"wm_eval_{i}"
            wm_samples.append(encoder.embed(_ref, wm_id))
            clean_samples.append(_ref)

        intensity_levels = [0.1, 0.3, 0.5] if self._test else [0.1, 0.2, 0.3, 0.4, 0.5]
        dilution_ratios = [0.5, 1.0] if self._test else [0.5, 1.0, 1.5, 2.0, 3.0]
        sub_budgets = [1, 5] if self._test else [1, 3, 5, 10, 15, 20]

        evasion_results: Dict[str, Any] = {}

        try:
            para_result = evaluator.evaluate_paraphrasing(
                wm_samples, clean_samples, intensity_levels=intensity_levels
            )
            evasion_results["paraphrase"] = para_result.summary()
        except Exception as exc:
            logger.log_error("evasion_paraphrase", exc, {})
            evasion_results["paraphrase"] = {"error": str(exc)}

        try:
            dilute_result = evaluator.evaluate_copy_paste_dilution(
                wm_samples, clean_samples, dilution_ratios=dilution_ratios
            )
            evasion_results["copy_paste_dilution"] = dilute_result.summary()
        except Exception as exc:
            logger.log_error("evasion_dilution", exc, {})
            evasion_results["copy_paste_dilution"] = {"error": str(exc)}

        try:
            sub_result = evaluator.evaluate_adaptive_substitution(
                wm_samples, clean_samples, substitution_budgets=sub_budgets
            )
            evasion_results["adaptive_substitution"] = sub_result.summary()
        except Exception as exc:
            logger.log_error("evasion_substitution", exc, {})
            evasion_results["adaptive_substitution"] = {"error": str(exc)}

        return evasion_results

    # ------------------------------------------------------------------
    # step 4: adaptive sad adversary
    # ------------------------------------------------------------------

    def _run_adaptive(self) -> Dict[str, Any]:
        """
        run adaptive adversary evaluation against sad for all attacks.

        returns dict mapping attack_type → AdaptiveSADResult dict.
        """
        from attacks.adaptive_attack import AdaptiveSADEvaluator

        n_trials = 1 if self._test else 2
        sigma_values = [1.5, 2.0, 2.5] if self._test else None

        logger.logger.info(
            "running adaptive adversary evaluation (n_trials=%d)", n_trials
        )
        evaluator = AdaptiveSADEvaluator(
            corpus_size=self.corpus_size,
            n_poison=self.n_poison,
            top_k=self.top_k,
            seed=self.seed_base,
        )

        results: Dict[str, Any] = {}
        for at in ["agent_poison", "minja", "injecmem"]:
            try:
                r = evaluator.evaluate(at, sigma_values=sigma_values, n_trials=n_trials)
                results[at] = r.to_dict()
                logger.logger.info(
                    "  %s: asr_r_std=%.3f asr_r_adv=%.3f evasion_rate=%.3f",
                    at,
                    r.asr_r_standard,
                    r.asr_r_adaptive,
                    r.evasion_rate,
                )
            except Exception as exc:
                logger.log_error("adaptive_eval", exc, {"attack": at})
                results[at] = {"error": str(exc)}

        return results

    # ------------------------------------------------------------------
    # step 5: ablation studies
    # ------------------------------------------------------------------

    def _run_ablations(self) -> Dict[str, Any]:
        """
        run all hyperparameter ablation studies.

        returns dict mapping study_name → list of AblationPoint dicts.
        """
        from evaluation.ablation_study import AblationStudy

        n_trials = 1 if self._test else 2
        logger.logger.info("running ablation studies (n_trials=%d)", n_trials)

        study = AblationStudy(seed_base=self.seed_base, n_bootstrap=200)
        raw_results = study.run_all(
            n_trials=n_trials,
            attack_types=["agent_poison", "minja", "injecmem"],
        )

        # serialize AblationPoint objects to dicts
        serialized: Dict[str, Any] = {}
        for key, pts in raw_results.items():
            if isinstance(pts, list):
                serialized[key] = [pt.to_dict() for pt in pts]
            else:
                serialized[key] = pts

        return serialized

    # ------------------------------------------------------------------
    # main run method
    # ------------------------------------------------------------------

    def run(self) -> ComprehensiveResult:
        """
        run all evaluation components and return a ComprehensiveResult.

        ordering: attack eval → matrix → evasion → adaptive → ablations.
        each component is independently error-handled so a failure in one
        does not prevent the others from running.

        returns:
            ComprehensiveResult with all results and metadata.
        """
        t_start = time.time()
        logger.logger.info(
            "starting comprehensive evaluation (corpus=%d, seeds=%d, test=%s)",
            self.corpus_size,
            self.n_seeds,
            self._test,
        )

        result = ComprehensiveResult(
            generated_at=datetime.utcnow().isoformat() + "Z",
            config={
                "corpus_size": self.corpus_size,
                "n_poison": self.n_poison,
                "top_k": self.top_k,
                "n_seeds": self.n_seeds,
                "seed_base": self.seed_base,
                "run_matrix": self.run_matrix,
                "run_evasion": self.run_evasion,
                "run_adaptive": self.run_adaptive,
                "run_ablations": self.run_ablations,
                "test_mode": self._test,
            },
        )

        # 1. attack evaluation (always runs)
        logger.logger.info("[1/5] multi-trial attack evaluation")
        result.attack_summaries = self._run_attack_evaluation()

        # 2. attack-defense matrix
        if self.run_matrix:
            logger.logger.info("[2/5] attack-defense matrix")
            result.matrix_result_dict = self._run_matrix()

        # 3. watermark evasion
        if self.run_evasion:
            logger.logger.info("[3/5] watermark evasion analysis")
            result.evasion_results = self._run_evasion()

        # 4. adaptive adversary against sad
        if self.run_adaptive:
            logger.logger.info("[4/5] adaptive adversary evaluation")
            result.adaptive_sad_results = self._run_adaptive()

        # 5. ablation studies
        if self.run_ablations:
            logger.logger.info("[5/5] hyperparameter ablation studies")
            result.ablation_results = self._run_ablations()

        result.elapsed_s = time.time() - t_start
        logger.logger.info(
            "comprehensive evaluation complete in %.1fs", result.elapsed_s
        )
        return result

    # ------------------------------------------------------------------
    # table generation
    # ------------------------------------------------------------------

    def generate_paper_tables(
        self,
        result: ComprehensiveResult,
        output_dir: str = "results/tables",
    ) -> Dict[str, str]:
        """
        generate all latex tables for the paper from a ComprehensiveResult.

        tables generated:
            table1_attack_results.tex  — attack asr-r/a/t/isr with bootstrap ci
            table2_defense_matrix.tex  — attack-defense matrix (asr-r + effectiveness)
            table3_defense_tpr_fpr.tex — tpr/fpr per defense per attack
            table4_evasion.tex         — watermark evasion results summary
            table5_adaptive_sad.tex    — adaptive adversary vs sad
            table_ablation_*.tex       — per-ablation tables

        args:
            result: ComprehensiveResult from run()
            output_dir: directory to write .tex files

        returns:
            dict mapping table_name → file path
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, str] = {}

        # table 1: attack results with bootstrap ci
        try:
            table1 = self._make_attack_table(result.attack_summaries)
            p = str(out / "table1_attack_results.tex")
            Path(p).write_text(table1)
            saved["table1_attack_results"] = p
        except Exception as exc:
            logger.log_error("gen_table1", exc, {})

        # table 2 + 3: defense matrix
        if result.matrix_result_dict:
            try:
                # reconstruct from dict is complex; use raw dict for table gen
                table2 = self._make_matrix_table(
                    result.matrix_result_dict, metric="asr_r"
                )
                table3 = self._make_matrix_table(
                    result.matrix_result_dict, metric="tpr_fpr"
                )
                p2 = str(out / "table2_defense_matrix.tex")
                p3 = str(out / "table3_defense_tpr_fpr.tex")
                Path(p2).write_text(table2)
                Path(p3).write_text(table3)
                saved["table2_defense_matrix"] = p2
                saved["table3_defense_tpr_fpr"] = p3
            except Exception as exc:
                logger.log_error("gen_table2_3", exc, {})

        # table 4: evasion results
        if result.evasion_results:
            try:
                table4 = self._make_evasion_table(result.evasion_results)
                p = str(out / "table4_evasion.tex")
                Path(p).write_text(table4)
                saved["table4_evasion"] = p
            except Exception as exc:
                logger.log_error("gen_table4", exc, {})

        # table 5: adaptive sad
        if result.adaptive_sad_results:
            try:
                table5 = self._make_adaptive_table(result.adaptive_sad_results)
                p = str(out / "table5_adaptive_sad.tex")
                Path(p).write_text(table5)
                saved["table5_adaptive_sad"] = p
            except Exception as exc:
                logger.log_error("gen_table5", exc, {})

        # ablation tables
        if result.ablation_results:
            try:
                ablation_tex = self._make_ablation_summary_table(
                    result.ablation_results
                )
                p = str(out / "table6_ablation.tex")
                Path(p).write_text(ablation_tex)
                saved["table6_ablation"] = p
            except Exception as exc:
                logger.log_error("gen_ablation_table", exc, {})

        logger.logger.info("generated %d latex tables in %s", len(saved), output_dir)
        return saved

    def _make_attack_table(self, summaries: Dict[str, Any]) -> str:
        """generate table 1: attack results with bootstrap ci."""
        attack_labels = {
            "agent_poison": "AgentPoison \\citep{chen2024agentpoison}",
            "minja": "MINJA \\citep{dong2025minja}",
            "injecmem": "InjecMEM \\citep{injecmem2026}",
        }
        # modelled asr-a values (from paper definitions)
        asr_a_vals = {"agent_poison": 0.68, "minja": 0.76, "injecmem": 0.57}

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Attack evaluation results on synthetic memory corpus",
            r"  ($N=200$, $k=5$, seed-averaged over 5 trials).",
            r"  ASR-A$^*$ is modelled from paper-reported values.",
            r"  95\% CI via percentile bootstrap (2000 replicates).}",
            r"\label{tab:attack_results}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Attack & ASR-R & ASR-A$^*$ & ASR-T & Benign Acc \\",
            r"\midrule",
        ]
        for at, label in attack_labels.items():
            s = summaries.get(at, {})
            asr_r = s.get("asr_r", {})
            asr_t = s.get("asr_t", {})
            ba = s.get("benign_accuracy", {})
            asr_a = asr_a_vals.get(at, 0.0)

            if isinstance(asr_r, dict):
                _lo = asr_r.get("lower", 0)
                _hi = asr_r.get("upper", 0)
                r_str = f"{asr_r.get('mean', 0):.3f}" f"$_{{[{_lo:.3f},{_hi:.3f}]}}$"
            else:
                r_str = f"{asr_r:.3f}"

            t_val = asr_t.get("mean", 0) if isinstance(asr_t, dict) else asr_t
            b_val = ba.get("mean", 0) if isinstance(ba, dict) else ba

            lines.append(
                f"{label} & {r_str} & "
                f"{asr_a:.2f} & "
                f"{t_val:.3f} & "
                f"{b_val:.3f} \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _make_matrix_table(
        self, matrix_dict: Dict[str, Any], metric: str = "asr_r"
    ) -> str:
        """generate attack × defense matrix table from serialized MatrixResult."""
        attacks = ["agent_poison", "minja", "injecmem"]
        defenses = [
            "watermark",
            "validation",
            "proactive",
            "composite",
            "semantic_anomaly",
        ]
        attack_labels = {
            "agent_poison": "AgentPoison",
            "minja": "MINJA",
            "injecmem": "InjecMEM",
        }
        defense_labels = {
            "watermark": "Watermark",
            "validation": "Validation",
            "proactive": "Proactive",
            "composite": "Composite",
            "semantic_anomaly": "SAD (Ours)",
        }

        results_raw = matrix_dict.get("results", {})

        if metric == "asr_r":
            caption = (
                "Attack success rate (ASR-R) under each defense "
                "(pre-ingestion filtering). Defense effectiveness = "
                "$1 - \\text{ASR-R}_{\\text{defended}}"
                " / \\text{ASR-R}_{\\text{baseline}}$."
            )
            label = "tab:attack_defense_matrix"
            col_headers = " & ".join(defense_labels.values())
            lines = [
                r"\begin{table*}[t]",
                r"\centering",
                r"\small",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                r"\begin{tabular}{l|c|" + "c" * len(defenses) + "}",
                r"\toprule",
                f"Attack & Baseline & {col_headers} \\\\",
                r"\midrule",
            ]
            for at in attacks:
                row_data = results_raw.get(at, {})
                baseline = None
                # try to get baseline from first defense result
                for def_key in defenses:
                    pair = row_data.get(def_key, {})
                    if pair and not baseline:
                        baseline = pair.get("asr_r_baseline", None)
                        break
                baseline_str = f"{baseline:.3f}" if baseline is not None else "--"
                row_parts = [attack_labels.get(at, at), baseline_str]
                for def_key in defenses:
                    pair = row_data.get(def_key, {})
                    if pair:
                        val = pair.get("asr_r_under_defense", 0)
                        row_parts.append(f"{val:.3f}")
                    else:
                        row_parts.append("--")
                lines.append(" & ".join(row_parts) + " \\\\")

        else:  # tpr_fpr table
            caption = (
                "Defense detection rates: TPR (true positive — poison flagged) and "
                "FPR (false positive — benign flagged) per (attack, defense) pair."
            )
            label = "tab:defense_tpr_fpr"
            col_headers = " & ".join(
                [f"\\multicolumn{{2}}{{c}}{{{defense_labels[d]}}}" for d in defenses]
            )
            lines = [
                r"\begin{table*}[t]",
                r"\centering",
                r"\small",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                r"\begin{tabular}{l" + "|cc" * len(defenses) + "}",
                r"\toprule",
                f"Attack & {col_headers} \\\\",
                " & " + " & ".join(["TPR & FPR"] * len(defenses)) + " \\\\",
                r"\midrule",
            ]
            for at in attacks:
                row_data = results_raw.get(at, {})
                row_parts = [attack_labels.get(at, at)]
                for def_key in defenses:
                    pair = row_data.get(def_key, {})
                    if pair:
                        tpr = pair.get("defense_tpr", 0)
                        fpr = pair.get("defense_fpr", 0)
                        row_parts.append(f"{tpr:.3f} & {fpr:.3f}")
                    else:
                        row_parts.append("-- & --")
                lines.append(" & ".join(row_parts) + " \\\\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
        return "\n".join(lines)

    def _make_evasion_table(self, evasion_results: Dict[str, Any]) -> str:
        """generate table 4: watermark evasion summary."""
        strategy_labels = {
            "paraphrase": "Paraphrase",
            "copy_paste_dilution": "Copy-Paste Dilution",
            "adaptive_substitution": "Adaptive Substitution",
        }
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Watermark evasion results. TPR before and after evasion, "
            r"and fraction of samples where evasion succeeded ($z < z_{\text{thr}}$).}",
            r"\label{tab:evasion}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Evasion Strategy & TPR Before & TPR After & Evasion Rate \\",
            r"\midrule",
        ]
        for key, label in strategy_labels.items():
            r = evasion_results.get(key, {})
            if "error" in r:
                lines.append(f"{label} & -- & -- & -- \\\\")
            else:
                lines.append(
                    f"{label} & "
                    f"{r.get('tpr_before', 0):.3f} & "
                    f"{r.get('tpr_after', 0):.3f} & "
                    f"{r.get('evasion_success_rate', 0):.3f} \\\\"
                )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _make_adaptive_table(self, adaptive_results: Dict[str, Any]) -> str:
        """generate table 5: adaptive adversary vs sad."""
        attack_labels = {
            "agent_poison": "AgentPoison",
            "minja": "MINJA",
            "injecmem": "InjecMEM",
        }
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Adaptive adversary evaluation against SAD. "
            r"The adversary applies greedy synonym substitution to craft passages "
            r"below the detection threshold ($k=2$). Evasion Rate = fraction of "
            r"poison entries below threshold; Retrieval Degradation = ASR-R loss.}",
            r"\label{tab:adaptive_sad}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Attack & \multicolumn{2}{c}{Standard} & "
            r"\multicolumn{2}{c}{Adaptive} & Evasion & Ret. Degrad. \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r" & ASR-R & SAD TPR & ASR-R & SAD TPR & Rate & $\Delta$ASR-R \\",
            r"\midrule",
        ]
        for at, label in attack_labels.items():
            r = adaptive_results.get(at, {})
            if "error" in r:
                lines.append(f"{label} & -- & -- & -- & -- & -- & -- \\\\")
            else:
                lines.append(
                    f"{label} & "
                    f"{r.get('asr_r_standard', 0):.3f} & "
                    f"{r.get('sad_tpr_standard', 0):.3f} & "
                    f"{r.get('asr_r_adaptive', 0):.3f} & "
                    f"{r.get('sad_tpr_adaptive', 0):.3f} & "
                    f"{r.get('evasion_rate', 0):.3f} & "
                    f"{r.get('retrieval_degradation', 0):.3f} \\\\"
                )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _make_ablation_summary_table(self, ablation_results: Dict[str, Any]) -> str:
        """generate condensed ablation summary table."""
        cs_pts = ablation_results.get("corpus_size", [])
        sad_pts = ablation_results.get("sad_sigma_agent_poison", [])
        wm_pts = ablation_results.get("watermark_z_threshold", [])
        tk_pts = ablation_results.get("top_k_agent_poison", [])

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Hyperparameter ablation summary.  "
            r"Each row corresponds to one setting of the ablated parameter.}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lc|lc|lcc|lcc}",
            r"\toprule",
            r"\multicolumn{2}{c|}{Corpus} & "
            r"\multicolumn{2}{c|}{Top-K} & "
            r"\multicolumn{3}{c|}{SAD $k$} & "
            r"\multicolumn{3}{c}{WM $z$-thr} \\",
            r"$N$ & ASR-R & $k$ & ASR-R & $k$ & TPR & FPR & $z$ & TPR & FPR \\",
            r"\midrule",
        ]
        max_rows = max(len(cs_pts), len(tk_pts), len(sad_pts), len(wm_pts), 1)
        for i in range(max_rows):
            row = []
            if i < len(cs_pts) and isinstance(cs_pts[i], dict):
                pt = cs_pts[i]
                v = (
                    int(pt["param_value"])
                    if pt["param_value"] == int(pt["param_value"])
                    else pt["param_value"]
                )
                row.append(f"{v} & {pt['asr_r_mean']:.3f}")
            else:
                row.append("& ")
            if i < len(tk_pts) and isinstance(tk_pts[i], dict):
                pt = tk_pts[i]
                v = (
                    int(pt["param_value"])
                    if pt["param_value"] == int(pt["param_value"])
                    else pt["param_value"]
                )
                row.append(f"{v} & {pt['asr_r_mean']:.3f}")
            else:
                row.append("& ")
            if i < len(sad_pts) and isinstance(sad_pts[i], dict):
                pt = sad_pts[i]
                row.append(
                    f"{pt['param_value']:.1f} & "
                    f"{pt['tpr_mean']:.3f} & {pt['fpr_mean']:.3f}"
                )
            else:
                row.append("& & ")
            if i < len(wm_pts) and isinstance(wm_pts[i], dict):
                pt = wm_pts[i]
                row.append(
                    f"{pt['param_value']:.1f} & "
                    f"{pt['tpr_mean']:.3f} & {pt['fpr_mean']:.3f}"
                )
            else:
                row.append("& & ")
            lines.append(" & ".join(row) + " \\\\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)
