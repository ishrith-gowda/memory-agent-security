"""
statistical evaluation utilities for memory agent security research.

this module provides rigorous multi-trial evaluation with bootstrap confidence
intervals, hypothesis testing, and latex table generation for paper-ready results.

methods follow standard practice for top-tier ml security venues:
- percentile bootstrap ci (efron & tibshirani, 1993)
- paired t-test and wilcoxon signed-rank for attack/defense comparison
- cohen's d for practical effect sizes
- booktabs-style latex tables with ±ci formatting

references:
- efron & tibshirani. "an introduction to the bootstrap." chapman & hall, 1993.
- cohen, j. "a power primer." psychological bulletin, 1992.
- paired evaluation protocol: see agentpoison (chen et al., neurips 2024)
  and minja (dong et al., neurips 2025).

all comments are lowercase.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# scipy is optional — used for t.sf and wilcoxon; fall back to approximation
try:
    from scipy import stats as _scipy_stats

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# bootstrap confidence intervals
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """
    result from a bootstrap ci computation.

    fields:
        mean: sample mean of original data
        lower: lower bound of (1-alpha) ci
        upper: upper bound of (1-alpha) ci
        ci_width: upper - lower
        std: sample standard deviation of original data
        n_samples: number of original samples
        n_bootstrap: number of bootstrap replicates used
        alpha: significance level (default 0.05 → 95% ci)
    """

    mean: float
    lower: float
    upper: float
    ci_width: float
    std: float
    n_samples: int
    n_bootstrap: int
    alpha: float

    def __str__(self) -> str:
        half = self.ci_width / 2
        return (
            f"{self.mean:.3f} ± {half:.3f} "
            f"(95% CI [{self.lower:.3f}, {self.upper:.3f}])"
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "ci_width": self.ci_width,
            "std": self.std,
        }


class BootstrapCI:
    """
    percentile bootstrap confidence intervals for asr and defense metrics.

    uses the percentile method (efron 1979): resample with replacement
    n_bootstrap times, compute statistic on each resample, take empirical
    quantiles as ci bounds.  suitable for bounded metrics like asr ∈ [0,1].
    """

    def __init__(self, n_bootstrap: int = 2000, seed: int = 42):
        """
        args:
            n_bootstrap: number of bootstrap replicates
            seed: rng seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self._rng = random.Random(seed)

    def compute(
        self,
        samples: List[float],
        alpha: float = 0.05,
    ) -> BootstrapResult:
        """
        compute percentile bootstrap ci for a list of scalar samples.

        args:
            samples: list of observed metric values (e.g. asr-r per trial)
            alpha: significance level (0.05 → 95% ci)

        returns:
            BootstrapResult with mean, lower, upper, etc.
        """
        n = len(samples)
        if n == 0:
            raise ValueError("samples list cannot be empty")

        # draw bootstrap replicates
        boot_means: List[float] = []
        for _ in range(self.n_bootstrap):
            resample = [self._rng.choice(samples) for _ in range(n)]
            boot_means.append(sum(resample) / n)

        boot_means.sort()
        lo_idx = int(self.n_bootstrap * alpha / 2)
        hi_idx = min(int(self.n_bootstrap * (1.0 - alpha / 2)), self.n_bootstrap - 1)

        lower = boot_means[lo_idx]
        upper = boot_means[hi_idx]

        mean_val = sum(samples) / n
        var = sum((x - mean_val) ** 2 for x in samples) / max(n - 1, 1)
        std_val = math.sqrt(var)

        return BootstrapResult(
            mean=mean_val,
            lower=lower,
            upper=upper,
            ci_width=upper - lower,
            std=std_val,
            n_samples=n,
            n_bootstrap=self.n_bootstrap,
            alpha=alpha,
        )


# ---------------------------------------------------------------------------
# hypothesis testing
# ---------------------------------------------------------------------------


@dataclass
class HypothesisTestResult:
    """
    result from a statistical hypothesis test comparing two methods.

    fields:
        test_name: "paired_ttest" or "wilcoxon"
        statistic: test statistic value
        p_value: two-tailed p-value
        cohens_d: effect size (only for paired t-test; None for wilcoxon)
        significant: True if p_value < alpha
        effect_size_label: "negligible" / "small" / "medium" / "large"
    """

    test_name: str
    statistic: float
    p_value: float
    cohens_d: Optional[float]
    significant: bool
    effect_size_label: str

    @staticmethod
    def _effect_label(d: Optional[float]) -> str:
        """cohen's d magnitude to verbal label (cohen 1992)."""
        if d is None:
            return "n/a"
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        return "large"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "cohens_d": self.cohens_d,
            "significant": self.significant,
            "effect_size_label": self.effect_size_label,
        }


def _approx_t_pvalue(t: float, df: int) -> float:
    """
    two-tailed p-value approximation via normal cdf (for scipy-free fallback).

    accurate for df > 5; use scipy when available for exact values.
    """
    if df <= 0:
        return 1.0
    z = abs(t)
    # normal approximation: better than nothing when scipy unavailable
    p_one = 0.5 * math.erfc(z / math.sqrt(2.0))
    return min(2.0 * p_one, 1.0)


class StatisticalHypothesisTester:
    """
    paired statistical tests for comparing attack or defense metrics.

    intended use: compare asr-r (or other metric) across two methods,
    each evaluated on the same n_trials seeds.  paired tests exploit the
    within-seed correlation to increase power.
    """

    def paired_ttest(
        self,
        a: List[float],
        b: List[float],
        alpha: float = 0.05,
    ) -> HypothesisTestResult:
        """
        paired t-test: H0: mean(a - b) = 0.

        args:
            a: metric values for method a (one per trial)
            b: metric values for method b (same trials)
            alpha: significance threshold

        returns:
            HypothesisTestResult
        """
        if len(a) != len(b):
            raise ValueError("paired t-test requires equal-length lists")
        n = len(a)
        if n < 2:
            raise ValueError("need at least 2 paired samples")

        diffs = [x - y for x, y in zip(a, b)]
        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1)
        std_d = math.sqrt(var_d)

        if std_d < 1e-12:
            # no variance in differences — t is undefined
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = mean_d / (std_d / math.sqrt(n))
            if _SCIPY_AVAILABLE:
                p_value = float(2.0 * _scipy_stats.t.sf(abs(t_stat), df=n - 1))
            else:
                p_value = _approx_t_pvalue(t_stat, n - 1)

        # cohen's d for paired samples = mean_diff / std_diff
        cohens_d = mean_d / std_d if std_d > 1e-12 else 0.0

        return HypothesisTestResult(
            test_name="paired_ttest",
            statistic=t_stat,
            p_value=p_value,
            cohens_d=cohens_d,
            significant=p_value < alpha,
            effect_size_label=HypothesisTestResult._effect_label(cohens_d),
        )

    def wilcoxon(
        self,
        a: List[float],
        b: List[float],
        alpha: float = 0.05,
    ) -> HypothesisTestResult:
        """
        wilcoxon signed-rank test (non-parametric paired alternative).

        requires scipy.  raises RuntimeError if not available.
        """
        if not _SCIPY_AVAILABLE:
            raise RuntimeError(
                "scipy is required for the wilcoxon test — "
                "install it with: pip install scipy"
            )
        diffs = [x - y for x, y in zip(a, b)]
        stat, p = _scipy_stats.wilcoxon(diffs)
        return HypothesisTestResult(
            test_name="wilcoxon",
            statistic=float(stat),
            p_value=float(p),
            cohens_d=None,
            significant=float(p) < alpha,
            effect_size_label="n/a",
        )

    def bonferroni_correct(
        self,
        p_values: List[float],
        alpha: float = 0.05,
    ) -> List[bool]:
        """
        bonferroni correction for multiple comparisons.

        args:
            p_values: list of raw p-values from individual tests
            alpha: family-wise error rate

        returns:
            list of booleans indicating significance after correction
        """
        corrected_alpha = alpha / len(p_values)
        return [p < corrected_alpha for p in p_values]


# ---------------------------------------------------------------------------
# multi-trial evaluator
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """single-trial metrics from RetrievalSimulator.evaluate_attack()."""

    seed: int
    attack_type: str
    asr_r: float
    asr_a: float
    asr_t: float
    benign_accuracy: float
    injection_success_rate: float
    elapsed_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "attack_type": self.attack_type,
            "asr_r": self.asr_r,
            "asr_a": self.asr_a,
            "asr_t": self.asr_t,
            "benign_accuracy": self.benign_accuracy,
            "injection_success_rate": self.injection_success_rate,
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class MultiTrialSummary:
    """
    aggregated summary over n trials with bootstrap confidence intervals.

    each bootstrap field (asr_r, asr_a, …) is a BootstrapResult containing
    the mean and 95% ci across n_trials independent runs.
    """

    attack_type: str
    n_trials: int
    corpus_size: int
    n_poison: int
    top_k: int
    trial_results: List[TrialResult]
    asr_r: Optional[BootstrapResult] = field(default=None)
    asr_a: Optional[BootstrapResult] = field(default=None)
    asr_t: Optional[BootstrapResult] = field(default=None)
    benign_accuracy: Optional[BootstrapResult] = field(default=None)
    isr: Optional[BootstrapResult] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        def _br(br: Optional[BootstrapResult]) -> Optional[Dict]:
            return br.to_dict() if br is not None else None

        return {
            "attack_type": self.attack_type,
            "n_trials": self.n_trials,
            "corpus_size": self.corpus_size,
            "n_poison": self.n_poison,
            "top_k": self.top_k,
            "asr_r": _br(self.asr_r),
            "asr_a": _br(self.asr_a),
            "asr_t": _br(self.asr_t),
            "benign_accuracy": _br(self.benign_accuracy),
            "isr": _br(self.isr),
            "trials": [r.to_dict() for r in self.trial_results],
        }


class MultiTrialEvaluator:
    """
    run retrieval simulator over multiple random seeds for statistical rigor.

    each trial uses a fresh RetrievalSimulator with different corpus shuffling
    (controlled by Python's random seed).  bootstrap cis are then computed
    across the n_trials scalar measurements.

    usage:
        evaluator = MultiTrialEvaluator(corpus_size=200, n_poison=5, top_k=5)
        summary = evaluator.evaluate_attack("minja", n_trials=5)
        print(summary.asr_r)  # BootstrapResult with mean ± ci
    """

    def __init__(
        self,
        corpus_size: int = 200,
        n_poison: int = 5,
        top_k: int = 5,
        use_trigger_optimization: bool = False,
        n_bootstrap: int = 1000,
    ):
        """
        args:
            corpus_size: number of benign entries in each trial corpus
            n_poison: number of adversarial entries to inject per trial
            top_k: retrieval top-k
            use_trigger_optimization: enable agentpoison trigger optimizer
            n_bootstrap: bootstrap replicates for ci computation
        """
        self.corpus_size = corpus_size
        self.n_poison = n_poison
        self.top_k = top_k
        self.use_trigger_optimization = use_trigger_optimization
        self._bootstrap = BootstrapCI(n_bootstrap=n_bootstrap)

    def evaluate_attack(
        self,
        attack_type: str,
        n_trials: int = 5,
        seeds: Optional[List[int]] = None,
    ) -> MultiTrialSummary:
        """
        evaluate one attack type over n_trials independent seeds.

        args:
            attack_type: "agent_poison", "minja", or "injecmem"
            n_trials: number of trials (ignored if seeds provided)
            seeds: explicit seed list; if None, uses range(n_trials)

        returns:
            MultiTrialSummary with per-trial results and bootstrap cis
        """
        from evaluation.retrieval_sim import RetrievalSimulator

        if seeds is None:
            seeds = list(range(n_trials))

        trial_results: List[TrialResult] = []
        for seed in seeds:
            random.seed(seed)
            sim = RetrievalSimulator(
                corpus_size=self.corpus_size,
                n_poison_per_attack=self.n_poison,
                top_k=self.top_k,
                use_trigger_optimization=self.use_trigger_optimization,
            )
            t0 = time.perf_counter()
            metrics = sim.evaluate_attack(attack_type)
            elapsed = time.perf_counter() - t0

            trial_results.append(
                TrialResult(
                    seed=seed,
                    attack_type=attack_type,
                    asr_r=metrics.asr_r,
                    asr_a=metrics.asr_a,
                    asr_t=metrics.asr_t,
                    benign_accuracy=metrics.benign_accuracy,
                    injection_success_rate=metrics.injection_success_rate,
                    elapsed_s=elapsed,
                )
            )

        ci = self._bootstrap
        summary = MultiTrialSummary(
            attack_type=attack_type,
            n_trials=len(trial_results),
            corpus_size=self.corpus_size,
            n_poison=self.n_poison,
            top_k=self.top_k,
            trial_results=trial_results,
        )
        summary.asr_r = ci.compute([r.asr_r for r in trial_results])
        summary.asr_a = ci.compute([r.asr_a for r in trial_results])
        summary.asr_t = ci.compute([r.asr_t for r in trial_results])
        summary.benign_accuracy = ci.compute([r.benign_accuracy for r in trial_results])
        summary.isr = ci.compute([r.injection_success_rate for r in trial_results])
        return summary

    def evaluate_all_attacks(self, n_trials: int = 5) -> Dict[str, MultiTrialSummary]:
        """evaluate all three attacks over n_trials seeds."""
        results: Dict[str, MultiTrialSummary] = {}
        for attack in ["agent_poison", "minja", "injecmem"]:
            results[attack] = self.evaluate_attack(attack, n_trials=n_trials)
        return results

    def compare_attacks(
        self,
        attack_a: str,
        attack_b: str,
        n_trials: int = 5,
        metric: str = "asr_r",
    ) -> HypothesisTestResult:
        """
        compare two attacks on a given metric via paired t-test.

        uses the same seeds for both attacks to enable paired testing.
        """
        seeds = list(range(n_trials))
        summary_a = self.evaluate_attack(attack_a, seeds=seeds)
        summary_b = self.evaluate_attack(attack_b, seeds=seeds)

        vals_a = [getattr(r, metric) for r in summary_a.trial_results]
        vals_b = [getattr(r, metric) for r in summary_b.trial_results]

        tester = StatisticalHypothesisTester()
        return tester.paired_ttest(vals_a, vals_b)


# ---------------------------------------------------------------------------
# latex table generator
# ---------------------------------------------------------------------------


class LatexTableGenerator:
    """
    generate paper-ready latex tables with booktabs formatting.

    follows conventions from top-tier ml venues:
    - \\toprule / \\midrule / \\bottomrule  (requires booktabs package)
    - \\textbf{...} for best result in each column
    - ±ci formatting: "0.350$\\pm$0.045"
    - multi-row column headers where appropriate
    """

    def __init__(self, bold_best: bool = True):
        """
        args:
            bold_best: whether to bold the best value per numeric column
        """
        self.bold_best = bold_best

    # ------------------------------------------------------------------
    # formatting helpers
    # ------------------------------------------------------------------

    def _fmt(self, val: float, ci: Optional[BootstrapResult] = None) -> str:
        """format a float with optional ±half-ci."""
        if ci is not None:
            half = ci.ci_width / 2.0
            return f"{val:.3f}$\\pm${half:.3f}"
        return f"{val:.3f}"

    @staticmethod
    def _bold(s: str) -> str:
        return f"\\textbf{{{s}}}"

    @staticmethod
    def _underline(s: str) -> str:
        return f"\\underline{{{s}}}"

    def _extract_val(self, s: str) -> float:
        """extract leading float from a formatted string like '0.350$\\pm$...'."""
        try:
            return float(s.split("$")[0])
        except (ValueError, IndexError):
            return 0.0

    def _apply_bold_best(
        self,
        rows: List[List[str]],
        col_indices: List[int],
        higher_is_better: bool = True,
    ) -> None:
        """in-place: bold the best value in each specified column."""
        for col_idx in col_indices:
            vals = [self._extract_val(row[col_idx]) for row in rows]
            if not vals:
                continue
            best = max(vals) if higher_is_better else min(vals)
            for i, row in enumerate(rows):
                if abs(vals[i] - best) < 1e-9:
                    row[col_idx] = self._bold(row[col_idx])

    # ------------------------------------------------------------------
    # table generators
    # ------------------------------------------------------------------

    def generate_attack_table(
        self,
        summaries: Dict[str, MultiTrialSummary],
        caption: str = (
            "Attack evaluation on synthetic agent memory corpus "
            "(200 entries, 5 poison, top-$k$=5). "
            "Results averaged over 5 random seeds with 95\\% bootstrap CIs."
        ),
        label: str = "tab:attacks",
    ) -> str:
        """
        generate attack comparison table (Table 1).

        columns: attack | asr-r | asr-a | asr-t | isr | benign acc
        rows: agentpoison, minja, injecmem
        """
        attack_display = {
            "agent_poison": "AgentPoison~\\cite{chen2024agentpoison}",
            "minja": "MINJA~\\cite{dong2025minja}",
            "injecmem": "InjecMEM~\\cite{injecmem2026}",
        }

        rows: List[List[str]] = []
        for attack_type in ["agent_poison", "minja", "injecmem"]:
            if attack_type not in summaries:
                continue
            s = summaries[attack_type]
            row = [
                attack_display.get(attack_type, attack_type),
                self._fmt(s.asr_r.mean, s.asr_r) if s.asr_r else "—",
                self._fmt(s.asr_a.mean, s.asr_a) if s.asr_a else "—",
                self._fmt(s.asr_t.mean, s.asr_t) if s.asr_t else "—",
                self._fmt(s.isr.mean, s.isr) if s.isr else "—",
                (
                    self._fmt(s.benign_accuracy.mean, s.benign_accuracy)
                    if s.benign_accuracy
                    else "—"
                ),
            ]
            rows.append(row)

        if self.bold_best and rows:
            # asr metrics: higher is better (attack effectiveness)
            self._apply_bold_best(rows, [1, 2, 3, 4], higher_is_better=True)
            # benign accuracy: higher is better (less collateral damage)
            self._apply_bold_best(rows, [5], higher_is_better=True)

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            (
                "\\multirow{2}{*}{Attack} & \\multicolumn{5}{c}{Attack Success Rate} "
                "& \\multirow{2}{*}{Benign Acc} \\\\"
            ),
            "\\cmidrule(lr){2-6}",
            "& ASR-R & ASR-A & ASR-T & ISR & (95\\% CI) \\\\",
            "\\midrule",
        ]
        for row in rows:
            lines.append(" & ".join(row) + " \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
        return "\n".join(lines)

    def generate_defense_table(
        self,
        defense_results: Dict[str, Dict[str, Any]],
        caption: str = (
            "Defense evaluation: detection rates on synthetic poisoned corpus. "
            "TPR = true positive rate, FPR = false positive rate."
        ),
        label: str = "tab:defenses",
    ) -> str:
        """
        generate defense comparison table (Table 2).

        columns: defense | tpr | fpr | f1 | auroc | latency
        """
        defense_display = {
            "watermark": "Unigram Watermark~\\cite{zhao2024unigram}",
            "validation": "Content Validation",
            "proactive": "Proactive Simulation",
            "composite": "Composite Defense",
            "semantic_anomaly": "SAD (ours)",
        }

        rows: List[List[str]] = []
        for defense_type, metrics in defense_results.items():
            name = defense_display.get(defense_type, defense_type)
            tpr = metrics.get("tpr", 0.0)
            fpr = metrics.get("fpr", 0.0)
            f1 = metrics.get("f1", metrics.get("f1_score", 0.0))
            auroc = metrics.get("auroc", 0.0)
            latency = metrics.get("latency_ms", 0.0)
            row = [
                name,
                f"{tpr:.3f}",
                f"{fpr:.3f}",
                f"{f1:.3f}",
                f"{auroc:.3f}",
                f"{latency:.1f}",
            ]
            rows.append(row)

        if self.bold_best and rows:
            # tpr, f1, auroc: higher is better
            self._apply_bold_best(rows, [1, 3, 4], higher_is_better=True)
            # fpr: lower is better
            self._apply_bold_best(rows, [2], higher_is_better=False)

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Defense & TPR$\\uparrow$ & FPR$\\downarrow$ & F1$\\uparrow$ "
            "& AUROC$\\uparrow$ & Latency (ms) \\\\",
            "\\midrule",
        ]
        for row in rows:
            lines.append(" & ".join(row) + " \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
        return "\n".join(lines)

    def generate_ablation_table(
        self,
        ablation_rows: List[Dict[str, Any]],
        param_name: str,
        metric_names: List[str],
        caption: str = "Ablation study",
        label: str = "tab:ablation",
    ) -> str:
        """
        generate ablation table varying one hyperparameter.

        args:
            ablation_rows: list of dicts, each with param_name + metric_names
            param_name: column header for the varied parameter
            metric_names: list of metric column headers
            caption: latex caption
            label: latex label
        """
        n_cols = 1 + len(metric_names)
        col_spec = "l" + "c" * len(metric_names)
        header = " & ".join([param_name] + metric_names)

        rows: List[List[str]] = []
        for row_data in ablation_rows:
            val = str(row_data.get(param_name, ""))
            metric_vals = [f"{row_data.get(m, 0.0):.3f}" for m in metric_names]
            rows.append([val] + metric_vals)

        if self.bold_best and rows:
            self._apply_bold_best(rows, list(range(1, n_cols)), higher_is_better=True)

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            header + " \\\\",
            "\\midrule",
        ]
        for row in rows:
            lines.append(" & ".join(row) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        return "\n".join(lines)

    def save(self, latex_str: str, output_path: str) -> str:
        """save latex string to file."""
        with open(output_path, "w") as f:
            f.write(latex_str)
        return output_path
