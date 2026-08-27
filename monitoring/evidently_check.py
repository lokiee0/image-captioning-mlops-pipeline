"""
Stage 8: Monitor - data drift check using Evidently AI.

Compares simple image-level feature stats (brightness, size, aspect ratio)
and caption-length distribution between the original training set (baseline)
and a sample of recent production requests (logged by the FastAPI service),
returning the share of drifted features. retrain_trigger_dag.py acts on this.
"""
import pandas as pd


def _load_baseline() -> pd.DataFrame:
    # In production this reads the feature snapshot saved alongside the
    # registered model version. Placeholder synthetic frame keeps this
    # script runnable standalone for testing.
    return pd.DataFrame(
        {
            "brightness": [120, 130, 110, 125, 118],
            "aspect_ratio": [1.33, 1.5, 1.0, 1.33, 1.78],
            "caption_length": [8, 10, 7, 9, 11],
        }
    )


def _load_recent_production_sample() -> pd.DataFrame:
    # In production this queries the request log table populated by
    # serving/app.py for the last N hours of traffic.
    return pd.DataFrame(
        {
            "brightness": [95, 88, 100, 92, 90],
            "aspect_ratio": [1.6, 1.9, 2.0, 1.7, 1.85],
            "caption_length": [5, 4, 6, 5, 5],
        }
    )


def run_drift_check() -> float:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    baseline = _load_baseline()
    current = _load_recent_production_sample()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline, current_data=current)
    result = report.as_dict()

    return result["metrics"][0]["result"]["share_of_drifted_columns"]


if __name__ == "__main__":
    share = run_drift_check()
    print(f"drifted column share: {share:.3f}")
