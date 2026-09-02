"""Audit saved R1Pro policy commands against realized 5 Hz compact-state deltas."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from openpi.weaver import b1k_dataset_adapter
from openpi.weaver import causal_action_adapter
from openpi.weaver.b1k_contract import build_r1pro_contract
from openpi.weaver.behavior_precomputed import load_numeric_episode

SCHEMA_VERSION = "b1k_weaver_action_semantics_audit_v2"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--episode-indices", type=int, nargs="+", required=True)
    parser.add_argument("--min-lag-frames", type=int, default=-12)
    parser.add_argument("--max-lag-frames", type=int, default=12)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.min_lag_frames > args.max_lag_frames:
        raise ValueError("--min-lag-frames must not exceed --max-lag-frames")
    if len(set(args.episode_indices)) != len(args.episode_indices) or any(
        index < 0 for index in args.episode_indices
    ):
        raise ValueError("--episode-indices must be unique non-negative integers")

    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    manifests = causal_action_adapter.read_manifest_index(dataset_root, split=args.split)
    by_index = {int(manifest["episode_index"]): manifest for manifest in manifests}
    missing = sorted(set(args.episode_indices) - set(by_index))
    if missing:
        raise ValueError(f"Requested episodes are absent from split {args.split!r}: {missing}")

    contract = build_r1pro_contract()
    lags = tuple(range(args.min_lag_frames, args.max_lag_frames + 1))
    predictors: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "raw_start",
            "raw_end",
            "raw_duration_mean",
            "openpi_start",
            "openpi_end",
            "openpi_duration_mean",
            "command_state_error_start",
            "command_state_error_end",
            "command_state_error_duration_mean",
        )
    }
    lag_predictors = {
        representation: {lag: [] for lag in lags}
        for representation in ("raw_point", "openpi_point", "command_state_error_point")
    }
    lag_targets = {lag: [] for lag in lags}
    targets = []
    rows = []
    episode_semantics = []

    for episode_index in args.episode_indices:
        manifest = by_index[episode_index]
        b1k_dataset_adapter.verify_episode_manifest(manifest)
        numeric = load_numeric_episode(dataset_root, manifest)
        episode_semantics.append(
            {
                "episode_index": episode_index,
                "semantic_sha256": manifest["semantic_sha256"],
                "sampled_count": numeric.length,
            }
        )
        for interval_index in range(numeric.length - 1):
            if not bool(numeric.transition_valid[interval_index]):
                continue
            start = int(numeric.source_frame_indices[interval_index])
            end = int(numeric.source_frame_indices[interval_index + 1])
            if not 0 <= start < end < len(numeric.source_timestamps_s):
                raise ValueError(f"Invalid source interval for episode {episode_index}: {start}:{end}")
            raw = np.asarray(numeric.source_policy_actions[start:end], dtype=np.float64)
            durations = np.diff(np.asarray(numeric.source_timestamps_s[start : end + 1], dtype=np.float64))
            current_state = np.asarray(numeric.states[interval_index], dtype=np.float64)
            target = np.asarray(numeric.actions[interval_index], dtype=np.float64)
            openpi = contract.environment_to_openpi_training_actions(raw, current_state)
            command_state_error = _command_minus_current_state(raw, current_state)
            duration_weight = durations / durations.sum()

            values = {
                "raw_start": raw[0],
                "raw_end": raw[-1],
                "raw_duration_mean": np.sum(raw * duration_weight[:, None], axis=0),
                "openpi_start": openpi[0],
                "openpi_end": openpi[-1],
                "openpi_duration_mean": np.sum(openpi * duration_weight[:, None], axis=0),
                "command_state_error_start": command_state_error[0],
                "command_state_error_end": command_state_error[-1],
                "command_state_error_duration_mean": np.sum(
                    command_state_error * duration_weight[:, None], axis=0
                ),
            }
            targets.append(target)
            for name, value in values.items():
                predictors[name].append(value)
            rows.append(
                {
                    "episode_index": episode_index,
                    "interval_index": interval_index,
                    "source_start": start,
                    "source_end": end,
                    "command_count": len(raw),
                    "duration_s": float(durations.sum()),
                }
            )

            for lag in lags:
                source_index = start + lag
                if not 0 <= source_index < len(numeric.source_policy_actions):
                    continue
                raw_point = np.asarray(numeric.source_policy_actions[source_index], dtype=np.float64)
                openpi_point = contract.environment_to_openpi_training_actions(raw_point[None], current_state)[0]
                command_state_error_point = _command_minus_current_state(raw_point[None], current_state)[0]
                lag_targets[lag].append(target)
                lag_predictors["raw_point"][lag].append(raw_point)
                lag_predictors["openpi_point"][lag].append(openpi_point)
                lag_predictors["command_state_error_point"][lag].append(command_state_error_point)

    target_array = np.stack(targets)
    dimension_labels = _dimension_labels(contract)
    representation_reports = {
        name: _report(np.stack(values), target_array, contract=contract, dimension_labels=dimension_labels)
        for name, values in predictors.items()
    }
    lag_reports = {}
    for representation, by_lag in lag_predictors.items():
        lag_reports[representation] = {
            str(lag): _report(
                np.stack(values),
                np.stack(lag_targets[lag]),
                contract=contract,
                dimension_labels=dimension_labels,
            )
            for lag, values in by_lag.items()
            if values
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "split": args.split,
        "episodes": episode_semantics,
        "transition_count": len(target_array),
        "source_interval": {
            "command_count_min": min(row["command_count"] for row in rows),
            "command_count_max": max(row["command_count"] for row in rows),
            "duration_s_min": min(row["duration_s"] for row in rows),
            "duration_s_max": max(row["duration_s"] for row in rows),
        },
        "action_components": [
            {
                "name": component.name,
                "indices": list(component.indices),
                "environment_semantics": component.environment_semantics,
                "openpi_training_semantics": component.openpi_training_semantics,
                "compact_state_indices": (
                    list(component.compact_state_indices) if component.compact_state_indices is not None else None
                ),
            }
            for component in contract.action_components
        ],
        "representations": representation_reports,
        "lag_scan": {
            "unit": "source_30hz_frames",
            "min": args.min_lag_frames,
            "max": args.max_lag_frames,
            "reports": lag_reports,
        },
        "artifact_identity": _artifact_identity(episode_semantics, args=args),
    }
    output_path = Path(args.output_json).expanduser().resolve()
    _atomic_write_json(output_path, payload)
    _print_summary(payload, output_path=output_path)


def _dimension_labels(contract: Any) -> list[dict[str, Any]]:
    state_by_index = {
        index: component.name
        for component in contract.state_components
        for index in component.compact_indices
    }
    action_by_index = {
        index: component.name
        for component in contract.action_components
        for index in component.indices
    }
    return [
        {
            "index": index,
            "action_component": action_by_index[index],
            "state_component": state_by_index[index],
        }
        for index in range(contract.action_dim)
    ]


def _command_minus_current_state(commands: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    actions = np.asarray(commands, dtype=np.float64)
    state = np.asarray(current_state, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 23 or state.shape != (23,):
        raise ValueError("command-state error requires commands (S,23) and current_state (23,)")
    if not np.isfinite(actions).all() or not np.isfinite(state).all():
        raise ValueError("command-state error inputs must be finite")
    return actions - state[None]


def _dimension_metrics(predictor: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Dimension metrics require matching finite vectors with at least two rows")
    x_mean, y_mean = float(x.mean()), float(y.mean())
    x_std, y_std = float(x.std()), float(y.std())
    residual = x - y
    correlation = None
    standardized_rmse = None
    if x_std > 0 and y_std > 0:
        correlation = float(np.corrcoef(x, y)[0, 1])
        standardized_rmse = float(np.sqrt(np.mean(((x - x_mean) / x_std - (y - y_mean) / y_std) ** 2)))
    design = np.column_stack([x, np.ones_like(x)])
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = scale * x + offset
    target_ss = float(np.sum((y - y_mean) ** 2))
    affine_r2 = None if target_ss == 0 else float(1.0 - np.sum((y - fitted) ** 2) / target_ss)
    origin_denominator = float(np.dot(x, x))
    origin_scale = float(np.dot(x, y) / origin_denominator) if origin_denominator > 0 else None
    origin_fitted = x * origin_scale if origin_scale is not None else np.zeros_like(y)
    rmse = float(np.sqrt(np.mean(residual**2)))
    origin_rmse = float(np.sqrt(np.mean((origin_fitted - y) ** 2)))
    return {
        "count": len(x),
        "predictor_mean": x_mean,
        "predictor_std": x_std,
        "target_mean": y_mean,
        "target_std": y_std,
        "correlation": correlation,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": rmse,
        "direct_nrmse": rmse / y_std if y_std > 0 else None,
        "standardized_rmse": standardized_rmse,
        "affine_scale": float(scale),
        "affine_offset": float(offset),
        "affine_r2": affine_r2,
        "origin_scale": origin_scale,
        "origin_rmse": origin_rmse,
        "origin_nrmse": origin_rmse / y_std if y_std > 0 else None,
    }


def _report(
    predictor: np.ndarray,
    target: np.ndarray,
    *,
    contract: Any,
    dimension_labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if predictor.shape != target.shape or predictor.ndim != 2 or predictor.shape[1] != contract.action_dim:
        raise ValueError(f"Predictor/target must share (N,{contract.action_dim}) shape")
    dimensions = []
    for label in dimension_labels:
        index = int(label["index"])
        dimensions.append({**label, **_dimension_metrics(predictor[:, index], target[:, index])})
    components = {}
    for component in contract.action_components:
        selected = [dimensions[index] for index in component.indices]
        correlations = [abs(value["correlation"]) for value in selected if value["correlation"] is not None]
        signed_correlations = [value["correlation"] for value in selected if value["correlation"] is not None]
        r2_values = [value["affine_r2"] for value in selected if value["affine_r2"] is not None]
        standardized = [value["standardized_rmse"] for value in selected if value["standardized_rmse"] is not None]
        direct_nrmse = [value["direct_nrmse"] for value in selected if value["direct_nrmse"] is not None]
        origin_nrmse = [value["origin_nrmse"] for value in selected if value["origin_nrmse"] is not None]
        components[component.name] = {
            "indices": list(component.indices),
            "count": len(predictor),
            "mean_correlation": float(np.mean(signed_correlations)) if signed_correlations else None,
            "mean_abs_correlation": float(np.mean(correlations)) if correlations else None,
            "mean_affine_r2": float(np.mean(r2_values)) if r2_values else None,
            "mean_direct_nrmse": float(np.mean(direct_nrmse)) if direct_nrmse else None,
            "mean_origin_nrmse": float(np.mean(origin_nrmse)) if origin_nrmse else None,
            "mean_standardized_rmse": float(np.mean(standardized)) if standardized else None,
            "high_correlation_dimensions": sum(value >= 0.8 for value in correlations),
            "high_positive_correlation_dimensions": sum(value >= 0.8 for value in signed_correlations),
            "high_negative_correlation_dimensions": sum(value <= -0.8 for value in signed_correlations),
            "dimension_count": len(component.indices),
        }
    return {"count": len(predictor), "components": components, "dimensions": dimensions}


def _artifact_identity(episodes: Sequence[Mapping[str, Any]], *, args: argparse.Namespace) -> str:
    value = {
        "schema_version": SCHEMA_VERSION,
        "split": args.split,
        "episodes": list(episodes),
        "min_lag_frames": args.min_lag_frames,
        "max_lag_frames": args.max_lag_frames,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _best_lag(report_by_lag: Mapping[str, Any], *, component: str) -> tuple[int, Mapping[str, Any]]:
    candidates = []
    for lag, report in report_by_lag.items():
        summary = report["components"][component]
        score = summary["mean_affine_r2"]
        if score is not None:
            candidates.append((float(score), -abs(int(lag)), int(lag), summary))
    if not candidates:
        raise ValueError(f"No finite lag metrics for component {component!r}")
    _, _, lag, summary = max(candidates)
    return lag, summary


def _print_summary(payload: Mapping[str, Any], *, output_path: Path) -> None:
    print(f"output={output_path}")
    print(f"transitions={payload['transition_count']} episodes={len(payload['episodes'])}")
    print("\n=== representation summaries ===")
    for representation, report in payload["representations"].items():
        print(representation)
        for component, value in report["components"].items():
            print(
                f"  {component:20s} corr={value['mean_correlation']!s:>8} "
                f"abs_corr={value['mean_abs_correlation']!s:>8} "
                f"affine_r2={value['mean_affine_r2']!s:>8} "
                f"direct_nrmse={value['mean_direct_nrmse']!s:>8} "
                f"origin_nrmse={value['mean_origin_nrmse']!s:>8} "
                f"high_corr=+{value['high_positive_correlation_dimensions']}"
                f"/-{value['high_negative_correlation_dimensions']}"
                f"/{value['dimension_count']}"
            )
    print("\n=== best point-command lag by affine R2 ===")
    for representation, reports in payload["lag_scan"]["reports"].items():
        print(representation)
        component_names = payload["representations"]["raw_start"]["components"]
        for component in component_names:
            lag, value = _best_lag(reports, component=component)
            print(
                f"  {component:20s} lag={lag:+d} "
                f"abs_corr={value['mean_abs_correlation']!s:>8} affine_r2={value['mean_affine_r2']!s:>8}"
            )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite action-semantics audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
