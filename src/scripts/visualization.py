"""
visualization scripts for memory agent security research.

this module provides:
- plotting functions for benchmark results
- performance analysis charts
- attack-defense comparison visualizations
- statistical analysis plots

all comments are lowercase.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.benchmarking import (AttackMetrics, BenchmarkResult,
                                     DefenseMetrics)
from utils.logging import logger


class BenchmarkVisualizer:
    """visualization class for benchmark results."""

    def __init__(self, output_dir: str = "reports/figures"):
        """initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        logger.log_visualization_start("benchmark_visualizer", str(self.output_dir))

    def plot_attack_success_rates(
        self, results: List[BenchmarkResult], save_path: Optional[str] = None
    ) -> plt.Figure:
        """plot attack success rates across different memory systems."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Attack Success Rates by Memory System and Attack Type", fontsize=16
        )

        attack_types = ["agent_poison", "minja", "injecmem"]
        memory_systems = ["mem0", "amem", "memgpt"]

        # Prepare data
        data = []
        for result in results:
            for attack_type, metrics in result.attack_metrics.items():
                for mem_sys in memory_systems:
                    if mem_sys in metrics.asr_r:
                        data.append(
                            {
                                "experiment": result.experiment_id,
                                "attack_type": attack_type,
                                "memory_system": mem_sys,
                                "asr_r": metrics.asr_r.get(mem_sys, 0),
                                "asr_a": metrics.asr_a.get(mem_sys, 0),
                                "asr_t": metrics.asr_t.get(mem_sys, 0),
                            }
                        )

        if not data:
            logger.log_visualization_error("no attack metrics data available")
            return fig

        df = pd.DataFrame(data)

        # Plot ASR-R (Retrieval)
        ax = axes[0, 0]
        sns.barplot(data=df, x="memory_system", y="asr_r", hue="attack_type", ax=ax)
        ax.set_title("ASR-R (Retrieval Success Rate)")
        ax.set_ylabel("Success Rate")
        ax.tick_params(axis="x", rotation=45)

        # Plot ASR-A (Availability)
        ax = axes[0, 1]
        sns.barplot(data=df, x="memory_system", y="asr_a", hue="attack_type", ax=ax)
        ax.set_title("ASR-A (Availability Success Rate)")
        ax.set_ylabel("Success Rate")
        ax.tick_params(axis="x", rotation=45)

        # Plot ASR-T (Tampering)
        ax = axes[1, 0]
        sns.barplot(data=df, x="memory_system", y="asr_t", hue="attack_type", ax=ax)
        ax.set_title("ASR-T (Tampering Success Rate)")
        ax.set_ylabel("Success Rate")
        ax.tick_params(axis="x", rotation=45)

        # Plot average across all metrics
        ax = axes[1, 1]
        df_avg = (
            df.groupby(["memory_system", "attack_type"])[["asr_r", "asr_a", "asr_t"]]
            .mean()
            .reset_index()
        )
        df_avg["overall_asr"] = df_avg[["asr_r", "asr_a", "asr_t"]].mean(axis=1)
        sns.barplot(
            data=df_avg, x="memory_system", y="overall_asr", hue="attack_type", ax=ax
        )
        ax.set_title("Overall ASR (Average)")
        ax.set_ylabel("Success Rate")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.log_visualization_save(save_path)

        return fig

    def plot_defense_effectiveness(
        self, results: List[BenchmarkResult], save_path: Optional[str] = None
    ) -> plt.Figure:
        """plot defense effectiveness metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Defense Effectiveness Metrics", fontsize=16)

        defense_types = ["watermark", "validation", "proactive", "composite"]

        # Prepare data
        data = []
        for result in results:
            for defense_type, metrics in result.defense_metrics.items():
                data.append(
                    {
                        "experiment": result.experiment_id,
                        "defense_type": defense_type,
                        "tpr": metrics.tpr,
                        "fpr": metrics.fpr,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "accuracy": (
                            (metrics.true_positives + metrics.true_negatives)
                            / metrics.total_tests
                            if metrics.total_tests > 0
                            else 0
                        ),
                    }
                )

        if not data:
            logger.log_visualization_error("no defense metrics data available")
            return fig

        df = pd.DataFrame(data)

        # Plot TPR vs FPR
        ax = axes[0, 0]
        sns.scatterplot(
            data=df,
            x="fpr",
            y="tpr",
            hue="defense_type",
            style="experiment",
            ax=ax,
            s=100,
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)  # Diagonal line
        ax.set_title("True Positive Rate vs False Positive Rate")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plot Precision-Recall
        ax = axes[0, 1]
        sns.scatterplot(
            data=df,
            x="recall",
            y="precision",
            hue="defense_type",
            style="experiment",
            ax=ax,
            s=100,
        )
        ax.set_title("Precision vs Recall")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plot accuracy by defense type
        ax = axes[1, 0]
        sns.boxplot(data=df, x="defense_type", y="accuracy", ax=ax)
        ax.set_title("Defense Accuracy Distribution")
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", rotation=45)

        # Plot performance comparison
        ax = axes[1, 1]
        metrics_to_plot = ["tpr", "precision", "accuracy"]
        df_melted = df.melt(
            id_vars=["defense_type"],
            value_vars=metrics_to_plot,
            var_name="metric",
            value_name="value",
        )
        sns.barplot(data=df_melted, x="defense_type", y="value", hue="metric", ax=ax)
        ax.set_title("Defense Performance Comparison")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.log_visualization_save(save_path)

        return fig

    def plot_performance_comparison(
        self, results: List[BenchmarkResult], save_path: Optional[str] = None
    ) -> plt.Figure:
        """plot performance comparison across experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Performance Analysis", fontsize=16)

        # Prepare data
        data = []
        for result in results:
            data.append(
                {
                    "experiment": result.experiment_id,
                    "duration": result.test_duration,
                    "memory_operations": result.total_memory_operations,
                    "integrity_score": result.memory_integrity_score,
                    "timestamp": datetime.fromtimestamp(result.timestamp),
                }
            )

        if not data:
            logger.log_visualization_error("no performance data available")
            return fig

        df = pd.DataFrame(data)

        # Plot test duration
        ax = axes[0, 0]
        sns.barplot(data=df, x="experiment", y="duration", ax=ax)
        ax.set_title("Test Duration by Experiment")
        ax.set_ylabel("Duration (seconds)")
        ax.tick_params(axis="x", rotation=45)

        # Plot memory operations
        ax = axes[0, 1]
        sns.barplot(data=df, x="experiment", y="memory_operations", ax=ax)
        ax.set_title("Memory Operations by Experiment")
        ax.set_ylabel("Number of Operations")
        ax.tick_params(axis="x", rotation=45)

        # Plot integrity scores
        ax = axes[1, 0]
        sns.barplot(data=df, x="experiment", y="integrity_score", ax=ax)
        ax.set_title("Memory Integrity Scores")
        ax.set_ylabel("Integrity Score")
        ax.tick_params(axis="x", rotation=45)

        # Plot timeline
        ax = axes[1, 1]
        df_sorted = df.sort_values("timestamp")
        ax.plot(df_sorted["timestamp"], df_sorted["duration"], "o-", label="Duration")
        ax.plot(
            df_sorted["timestamp"],
            df_sorted["memory_operations"] / 100,
            "s-",
            label="Operations (x100)",
        )
        ax.plot(
            df_sorted["timestamp"],
            df_sorted["integrity_score"] * 100,
            "^-",
            label="Integrity (%)",
        )
        ax.set_title("Performance Timeline")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Metrics")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.log_visualization_save(save_path)

        return fig

    def plot_attack_defense_heatmap(
        self, results: List[BenchmarkResult], save_path: Optional[str] = None
    ) -> plt.Figure:
        """create heatmap of attack-defense interactions."""
        fig, ax = plt.subplots(figsize=(12, 8))

        attack_types = ["agent_poison", "minja", "injecmem"]
        defense_types = ["watermark", "validation", "proactive", "composite"]

        # Prepare data matrix
        heatmap_data = np.zeros((len(defense_types), len(attack_types)))

        for result in results:
            for i, defense in enumerate(defense_types):
                if defense in result.defense_metrics:
                    defense_metrics = result.defense_metrics[defense]
                    for j, attack in enumerate(attack_types):
                        if attack in result.attack_metrics:
                            attack_metrics = result.attack_metrics[attack]
                            # Use TPR - FPR as effectiveness score
                            effectiveness = defense_metrics.tpr - defense_metrics.fpr
                            heatmap_data[i, j] = max(0, min(1, effectiveness))

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            xticklabels=attack_types,
            yticklabels=defense_types,
            ax=ax,
        )

        ax.set_title("Attack-Defense Effectiveness Heatmap")
        ax.set_xlabel("Attack Types")
        ax.set_ylabel("Defense Types")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.log_visualization_save(save_path)

        return fig

    def generate_comprehensive_report(
        self, results: List[BenchmarkResult], output_prefix: str = "benchmark_report"
    ) -> Dict[str, str]:
        """generate comprehensive visualization report."""
        saved_files = {}

        # Generate all plots
        plots = [
            ("attack_success_rates", self.plot_attack_success_rates),
            ("defense_effectiveness", self.plot_defense_effectiveness),
            ("performance_comparison", self.plot_performance_comparison),
            ("attack_defense_heatmap", self.plot_attack_defense_heatmap),
        ]

        for plot_name, plot_func in plots:
            save_path = self.output_dir / f"{output_prefix}_{plot_name}.png"
            try:
                plot_func(results, str(save_path))
                saved_files[plot_name] = str(save_path)
                logger.log_visualization_complete(plot_name, str(save_path))
            except Exception as e:
                logger.log_visualization_error(f"failed to generate {plot_name}: {e}")

        return saved_files


class StatisticalAnalyzer:
    """statistical analysis for benchmark results."""

    def __init__(self):
        """initialize statistical analyzer."""
        self.results_cache = {}

    def analyze_attack_patterns(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """analyze attack success patterns."""
        analysis = {
            "attack_effectiveness": {},
            "memory_system_vulnerability": {},
            "temporal_patterns": {},
            "statistical_significance": {},
        }

        # Aggregate attack metrics
        attack_data = {}
        for result in results:
            for attack_type, metrics in result.attack_metrics.items():
                if attack_type not in attack_data:
                    attack_data[attack_type] = []
                attack_data[attack_type].append(
                    {
                        "asr_r": metrics.asr_r,
                        "asr_a": metrics.asr_a,
                        "asr_t": metrics.asr_t,
                        "execution_time": metrics.execution_time_avg,
                    }
                )

        # Calculate statistics for each attack type
        for attack_type, data_list in attack_data.items():
            asr_r_values = [
                d["asr_r"] for d in data_list if isinstance(d["asr_r"], (int, float))
            ]
            asr_a_values = [
                d["asr_a"] for d in data_list if isinstance(d["asr_a"], (int, float))
            ]
            asr_t_values = [
                d["asr_t"] for d in data_list if isinstance(d["asr_t"], (int, float))
            ]

            analysis["attack_effectiveness"][attack_type] = {
                "asr_r_mean": np.mean(asr_r_values) if asr_r_values else 0,
                "asr_r_std": np.std(asr_r_values) if asr_r_values else 0,
                "asr_a_mean": np.mean(asr_a_values) if asr_a_values else 0,
                "asr_a_std": np.std(asr_a_values) if asr_a_values else 0,
                "asr_t_mean": np.mean(asr_t_values) if asr_t_values else 0,
                "asr_t_std": np.std(asr_t_values) if asr_t_values else 0,
                "sample_size": len(data_list),
            }

        return analysis

    def analyze_defense_robustness(
        self, results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """analyze defense robustness patterns."""
        analysis = {
            "defense_effectiveness": {},
            "false_positive_analysis": {},
            "performance_tradeoffs": {},
            "optimal_configurations": {},
        }

        # Aggregate defense metrics
        defense_data = {}
        for result in results:
            for defense_type, metrics in result.defense_metrics.items():
                if defense_type not in defense_data:
                    defense_data[defense_type] = []
                defense_data[defense_type].append(
                    {
                        "tpr": metrics.tpr,
                        "fpr": metrics.fpr,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                    }
                )

        # Calculate statistics for each defense type
        for defense_type, data_list in defense_data.items():
            tpr_values = [
                d["tpr"] for d in data_list if isinstance(d["tpr"], (int, float))
            ]
            fpr_values = [
                d["fpr"] for d in data_list if isinstance(d["fpr"], (int, float))
            ]
            precision_values = [
                d["precision"]
                for d in data_list
                if isinstance(d["precision"], (int, float))
            ]
            recall_values = [
                d["recall"] for d in data_list if isinstance(d["recall"], (int, float))
            ]

            analysis["defense_effectiveness"][defense_type] = {
                "tpr_mean": np.mean(tpr_values) if tpr_values else 0,
                "tpr_std": np.std(tpr_values) if tpr_values else 0,
                "fpr_mean": np.mean(fpr_values) if fpr_values else 0,
                "fpr_std": np.std(fpr_values) if fpr_values else 0,
                "precision_mean": np.mean(precision_values) if precision_values else 0,
                "recall_mean": np.mean(recall_values) if recall_values else 0,
                "sample_size": len(data_list),
            }

        return analysis

    def generate_statistical_report(
        self, results: List[BenchmarkResult], output_path: str
    ) -> str:
        """generate statistical analysis report."""
        attack_analysis = self.analyze_attack_patterns(results)
        defense_analysis = self.analyze_defense_robustness(results)

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(results),
            "attack_analysis": attack_analysis,
            "defense_analysis": defense_analysis,
            "summary": {
                "most_effective_attack": (
                    max(
                        attack_analysis["attack_effectiveness"].items(),
                        key=lambda x: x[1]["asr_r_mean"],
                    )[0]
                    if attack_analysis["attack_effectiveness"]
                    else None
                ),
                "most_robust_defense": (
                    max(
                        defense_analysis["defense_effectiveness"].items(),
                        key=lambda x: x[1]["tpr_mean"] - x[1]["fpr_mean"],
                    )[0]
                    if defense_analysis["defense_effectiveness"]
                    else None
                ),
            },
        }

        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path


def create_experiment_dashboard(
    results: List[BenchmarkResult], output_dir: str = "reports/dashboard"
) -> str:
    """create an interactive dashboard for experiment results."""
    dashboard_dir = Path(output_dir)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    visualizer = BenchmarkVisualizer(str(dashboard_dir))
    saved_plots = visualizer.generate_comprehensive_report(results, "dashboard")

    # Generate statistical analysis
    analyzer = StatisticalAnalyzer()
    stats_report = analyzer.generate_statistical_report(
        results, str(dashboard_dir / "statistical_analysis.json")
    )

    # Create HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Memory Agent Security Research Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .plot {{ margin: 20px 0; }}
            .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            h1, h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Memory Agent Security Research Dashboard</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Experiments: {len(results)}</p>
        
        <h2>Attack Success Rates</h2>
        <div class="plot">
            <img src="dashboard_attack_success_rates.png" alt="Attack Success Rates" style="max-width: 100%;">
        </div>
        
        <h2>Defense Effectiveness</h2>
        <div class="plot">
            <img src="dashboard_defense_effectiveness.png" alt="Defense Effectiveness" style="max-width: 100%;">
        </div>
        
        <h2>Performance Comparison</h2>
        <div class="plot">
            <img src="dashboard_performance_comparison.png" alt="Performance Comparison" style="max-width: 100%;">
        </div>
        
        <h2>Attack-Defense Interactions</h2>
        <div class="plot">
            <img src="dashboard_attack_defense_heatmap.png" alt="Attack-Defense Heatmap" style="max-width: 100%;">
        </div>
        
        <h2>Statistical Analysis</h2>
        <div class="stats">
            <p>Detailed statistical analysis available in: <a href="statistical_analysis.json">statistical_analysis.json</a></p>
        </div>
    </body>
    </html>
    """

    dashboard_path = dashboard_dir / "index.html"
    with open(dashboard_path, "w") as f:
        f.write(html_content)

    return str(dashboard_path)


if __name__ == "__main__":
    # example usage
    print("memory agent security visualization module")
    print("run with benchmark results to generate visualizations")
