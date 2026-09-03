from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile
from typing import Any

SCHEMA_VERSION = "b1k_weaver_prediction_comparison_v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fresh and finetuned WEAVER held-out state/latent prediction summaries."
    )
    parser.add_argument("--fresh-summary", required=True)
    parser.add_argument("--trained-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_summary(path_value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read WEAVER evaluation summary {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "b1k_weaver_prediction_eval_v1":
        raise ValueError(f"Not a WEAVER prediction evaluation summary: {path}")
    return path, payload


def _evaluation_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = summary.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("Evaluation summary has no evaluation identity")
    episode_indices = evaluation.get("episode_indices")
    if not isinstance(episode_indices, list) or not episode_indices:
        raise ValueError("Evaluation summary has no explicit held-out episode_indices")
    return {
        "split": summary.get("split"),
        "episode_indices": [int(value) for value in episode_indices],
        "horizons": [int(value) for value in summary.get("horizons", ())],
        "window_count": int(summary.get("window_count", 0)),
        "prediction_record_count": int(summary.get("prediction_record_count", 0)),
    }


def _finite_metric(container: Mapping[str, Any], name: str) -> float:
    value = container.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Metric {name!r} is absent or non-numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"Metric {name!r} is non-finite")
    return result


def _delta(fresh: float, trained: float) -> dict[str, Any]:
    return {
        "fresh": fresh,
        "trained": trained,
        "delta": trained - fresh,
        "relative": None if fresh == 0.0 else (trained - fresh) / abs(fresh),
        "improved": trained < fresh,
    }


def compare_summaries(fresh: Mapping[str, Any], trained: Mapping[str, Any]) -> dict[str, Any]:
    fresh_identity = _evaluation_identity(fresh)
    trained_identity = _evaluation_identity(trained)
    if fresh_identity != trained_identity:
        raise ValueError(
            "Fresh and trained evaluations do not cover the exact same held-out windows: "
            f"fresh={fresh_identity}, trained={trained_identity}"
        )

    fresh_metrics = fresh.get("metrics")
    trained_metrics = trained.get("metrics")
    if not isinstance(fresh_metrics, Mapping) or not isinstance(trained_metrics, Mapping):
        raise ValueError("Both summaries must contain metrics mappings")
    fresh_state = fresh_metrics.get("canonical_state")
    trained_state = trained_metrics.get("canonical_state")
    if not isinstance(fresh_state, Mapping) or not isinstance(trained_state, Mapping):
        raise ValueError("Both summaries must contain canonical_state metrics")

    state: dict[str, Any] = {}
    for horizon in fresh_identity["horizons"]:
        key = str(horizon)
        if key not in fresh_state or key not in trained_state:
            raise ValueError(f"Missing canonical_state horizon {horizon}")
        fresh_overall = fresh_state[key].get("overall", {})
        trained_overall = trained_state[key].get("overall", {})
        state[key] = {
            "mae": _delta(_finite_metric(fresh_overall, "mae"), _finite_metric(trained_overall, "mae")),
            "rmse": _delta(_finite_metric(fresh_overall, "rmse"), _finite_metric(trained_overall, "rmse")),
            "trained_vs_persistence_rmse": _delta(
                _finite_metric(trained_overall, "persistence_rmse"),
                _finite_metric(trained_overall, "rmse"),
            ),
        }

    latent: dict[str, Any] = {}
    fresh_latent = fresh_metrics.get("latent")
    trained_latent = trained_metrics.get("latent")
    if not isinstance(fresh_latent, Mapping) or not isinstance(trained_latent, Mapping):
        raise ValueError("Both summaries must contain latent metrics")
    for horizon in fresh_identity["horizons"]:
        key = str(horizon)
        fresh_views = fresh_latent.get(key)
        trained_views = trained_latent.get(key)
        if not isinstance(fresh_views, Mapping) or not isinstance(trained_views, Mapping):
            raise ValueError(f"Missing latent horizon {horizon}")
        if set(fresh_views) != set(trained_views) or not fresh_views:
            raise ValueError(f"Latent views differ at horizon {horizon}")
        latent[key] = {
            view: {
                "mse": _delta(
                    _finite_metric(fresh_views[view]["overall"], "mse"),
                    _finite_metric(trained_views[view]["overall"], "mse"),
                ),
                "rmse": _delta(
                    _finite_metric(fresh_views[view]["overall"], "rmse"),
                    _finite_metric(trained_views[view]["overall"], "rmse"),
                ),
            }
            for view in sorted(fresh_views)
        }

    comparisons = [entry for horizon in state.values() for entry in (horizon["mae"], horizon["rmse"])]
    comparisons.extend(
        entry for horizon in latent.values() for view in horizon.values() for entry in (view["mse"], view["rmse"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_identity": fresh_identity,
        "canonical_state": state,
        "latent": latent,
        "aggregate": {
            "metrics_compared": len(comparisons),
            "trained_wins": sum(bool(entry["improved"]) for entry in comparisons),
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite comparison without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    fresh_path, fresh = _load_summary(args.fresh_summary)
    trained_path, trained = _load_summary(args.trained_summary)
    result = compare_summaries(fresh, trained)
    result["inputs"] = {"fresh": str(fresh_path), "trained": str(trained_path)}
    _atomic_write_json(Path(args.output_json), result, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
