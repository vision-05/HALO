"""Singleton-style metrics collection for simulation runs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import matplotlib

# Headless backend: SimPy runs in a worker thread (e.g. FastAPI stream); macOS GUI
# backend (MacOSX) crashes with NSWindow off the main thread.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NegotiationEvent:
    timestamp: float
    scenario: str
    device_id: str
    participants: list[str]
    iterations: int
    converged: bool
    final_value: float
    satisfaction_scores: dict[str, float]
    carbon_intensity: float
    fallback_used: bool
    participant_preferences: dict[str, float] | None = None


@dataclass
class FailureEvent:
    timestamp: float
    device_id: str
    failure_type: str
    recovery_attempts: int
    recovery_succeeded: bool
    time_in_failed_state: float


@dataclass
class LearningEvent:
    timestamp: float
    person_id: str
    device_type: str
    ema_value: float
    bayesian_mu: float
    bayesian_sigma: float
    routine_stable: bool


class MetricsCollector:
    """Collects events; produces CSVs and plots after a scenario run."""

    def __init__(self, scenario_name: str, output_dir: str | None = None) -> None:
        self.scenario_name = scenario_name
        base = output_dir or os.path.join(os.path.dirname(__file__), "..", "outputs")
        self.output_dir = os.path.abspath(base)
        os.makedirs(self.output_dir, exist_ok=True)

        self.negotiation_events: list[NegotiationEvent] = []
        self.failure_events: list[FailureEvent] = []
        self.learning_events: list[LearningEvent] = []
        self._message_log: list[dict[str, Any]] = []

    def log_negotiation(self, event: NegotiationEvent) -> None:
        self.negotiation_events.append(event)

    def log_failure(self, event: FailureEvent) -> None:
        self.failure_events.append(event)

    def log_learning(self, event: LearningEvent) -> None:
        self.learning_events.append(event)

    def log_message_routed(self, record: dict[str, Any]) -> None:
        self._message_log.append(record)

    def negotiation_dataframe(self) -> pd.DataFrame:
        if not self.negotiation_events:
            return pd.DataFrame()
        rows = []
        for e in self.negotiation_events:
            row = {
                "timestamp": e.timestamp,
                "scenario": e.scenario,
                "device_id": e.device_id,
                "participants": ",".join(e.participants),
                "iterations": e.iterations,
                "converged": e.converged,
                "final_value": e.final_value,
                "carbon_intensity": e.carbon_intensity,
                "fallback_used": e.fallback_used,
            }
            for k, v in e.satisfaction_scores.items():
                row[f"satisfaction_{k}"] = v
            if e.participant_preferences:
                for k, v in e.participant_preferences.items():
                    row[f"preference_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def failure_dataframe(self) -> pd.DataFrame:
        if not self.failure_events:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "device_id",
                    "failure_type",
                    "recovery_attempts",
                    "recovery_succeeded",
                    "time_in_failed_state",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "timestamp": e.timestamp,
                    "device_id": e.device_id,
                    "failure_type": e.failure_type,
                    "recovery_attempts": e.recovery_attempts,
                    "recovery_succeeded": e.recovery_succeeded,
                    "time_in_failed_state": e.time_in_failed_state,
                }
                for e in self.failure_events
            ]
        )

    def save_outputs(self) -> tuple[str, str, str]:
        """Write CSVs and plot; return paths (negotiations_csv, failures_csv, plot_png)."""
        safe = self.scenario_name.replace("/", "_")
        neg_path = os.path.join(self.output_dir, f"{safe}_negotiations.csv")
        fail_path = os.path.join(self.output_dir, f"{safe}_failures.csv")
        plot_path = os.path.join(self.output_dir, f"{safe}_results.png")

        df_neg = self.negotiation_dataframe()
        df_neg.to_csv(neg_path, index=False)
        self.failure_dataframe().to_csv(fail_path, index=False)

        self._save_plot(plot_path, df_neg)
        logger.info("Wrote outputs: %s, %s, %s", neg_path, fail_path, plot_path)
        return neg_path, fail_path, plot_path

    def _save_plot(self, plot_path: str, df_neg: pd.DataFrame) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
        if df_neg.empty:
            for ax in axes:
                ax.text(0.5, 0.5, "No negotiation data", ha="center", va="center")
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            return

        ax0, ax1, ax2 = axes
        ax0.plot(df_neg["timestamp"], df_neg["iterations"], marker="o", markersize=3)
        ax0.set_title("Negotiation iterations over time")
        ax0.set_xlabel("Simulated time (min)")
        ax0.set_ylabel("Iterations")

        sat_cols = [c for c in df_neg.columns if c.startswith("satisfaction_")]
        for c in sat_cols:
            ax1.plot(df_neg["timestamp"], df_neg[c], label=c.replace("satisfaction_", ""))
        ax1.set_title("Satisfaction scores over time")
        ax1.set_xlabel("Simulated time (min)")
        ax1.set_ylabel("Score")
        if sat_cols:
            ax1.legend()

        ax2.plot(df_neg["timestamp"], df_neg["final_value"], label="Thermostat setpoint", color="black")
        pref_cols = [c for c in df_neg.columns if c.startswith("preference_")]
        for c in pref_cols:
            ax2.plot(df_neg["timestamp"], df_neg[c], label=c.replace("preference_", "pref "), alpha=0.7)
        ax2.set_title("Thermostat setpoint vs preferences over time")
        ax2.set_xlabel("Simulated time (min)")
        ax2.set_ylabel("°C")
        ax2.legend()

        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

    def summary_stats(self) -> dict[str, Any]:
        n = len(self.negotiation_events)
        converged = sum(1 for e in self.negotiation_events if e.converged)
        rate = (converged / n * 100) if n else 0.0
        iters = [e.iterations for e in self.negotiation_events if e.converged]
        mean_it = sum(iters) / len(iters) if iters else 0.0
        all_sat: list[float] = []
        for e in self.negotiation_events:
            all_sat.extend(e.satisfaction_scores.values())
        mean_sat = sum(all_sat) / len(all_sat) if all_sat else 0.0
        fails = len(self.failure_events)
        rec_ok = sum(1 for e in self.failure_events if e.recovery_succeeded)
        rec_rate = (rec_ok / fails * 100) if fails else 0.0
        return {
            "total_negotiations": n,
            "convergence_rate_pct": rate,
            "mean_iterations": mean_it,
            "mean_satisfaction": mean_sat,
            "total_failures": fails,
            "recovery_rate_pct": rec_rate,
        }
