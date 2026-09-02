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

SCHEMA_VERSION = "b1k_weaver_action_semantics_audit_v3"
PHASES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--fit-episode-indices", type=int, nargs="+")
    parser.add_argument("--held-out-episode-indices", type=int, nargs="+")
    parser.add_argument("--min-lag-frames", type=int, default=-12)
    parser.add_argument("--max-lag-frames", type=int, default=12)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.fit_episode_indices is not None or args.held_out_episode_indices is not None:
        _main_causal_gate(args)
        return
    if args.episode_indices is None:
        raise ValueError("Provide --episode-indices, or both fit and held-out episode indices")
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


def _main_causal_gate(args: argparse.Namespace) -> None:
    if not args.fit_episode_indices or not args.held_out_episode_indices:
        raise ValueError("Causal gate requires both --fit-episode-indices and --held-out-episode-indices")
    fit_ids = tuple(args.fit_episode_indices)
    held_ids = tuple(args.held_out_episode_indices)
    if set(fit_ids) & set(held_ids):
        raise ValueError("Fit and held-out episodes must be disjoint")
    requested = fit_ids + held_ids
    if len(set(requested)) != len(requested) or any(index < 0 for index in requested):
        raise ValueError("Episode indices must be unique non-negative integers")

    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    manifests = causal_action_adapter.read_manifest_index(dataset_root, split=args.split)
    by_index = {int(manifest["episode_index"]): manifest for manifest in manifests}
    missing = sorted(set(requested) - set(by_index))
    if missing:
        raise ValueError(f"Requested episodes are absent from split {args.split!r}: {missing}")
    contract = build_r1pro_contract()
    episodes = {
        episode_id: _collect_gate_episode(dataset_root, by_index[episode_id], contract=contract)
        for episode_id in requested
    }
    fit = _concatenate_gate_episodes(episodes, fit_ids)
    held = _concatenate_gate_episodes(episodes, held_ids)
    fitted = _fit_component_phase_affine(fit, contract=contract)
    evaluation = {
        "fit": _evaluate_frozen_adapter(fit, fitted, contract=contract),
        "held_out": _evaluate_frozen_adapter(held, fitted, contract=contract),
        "held_out_by_episode": {
            str(episode_id): _evaluate_frozen_adapter(episodes[episode_id], fitted, contract=contract)
            for episode_id in held_ids
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "split": args.split,
        "phase_definition": {
            "unit": "fraction_of_current_source_interval",
            "values": list(PHASES),
            "index_rule": "round(phase * (command_count - 1))",
            "interval_local": True,
        },
        "fit_episode_indices": list(fit_ids),
        "held_out_episode_indices": list(held_ids),
        "fitted_adapter": fitted,
        "evaluation": evaluation,
        "gripper_events": {
            "fit": _gripper_event_report(episodes, fit_ids, contract=contract),
            "held_out": _gripper_event_report(episodes, held_ids, contract=contract),
        },
    }
    output_path = Path(args.output_json).expanduser().resolve()
    _atomic_write_json(output_path, payload)
    _print_causal_gate(payload, output_path=output_path)


def _collect_gate_episode(dataset_root: Path, manifest: Mapping[str, Any], *, contract: Any) -> dict[str, Any]:
    b1k_dataset_adapter.verify_episode_manifest(manifest)
    numeric = load_numeric_episode(dataset_root, manifest)
    phase_values = {phase: [] for phase in PHASES}
    targets, interval_indices, end_commands, states = [], [], [], []
    for interval_index in range(numeric.length - 1):
        if not bool(numeric.transition_valid[interval_index]):
            continue
        start = int(numeric.source_frame_indices[interval_index])
        end = int(numeric.source_frame_indices[interval_index + 1])
        raw = np.asarray(numeric.source_policy_actions[start:end], dtype=np.float64)
        if not len(raw):
            continue
        state = np.asarray(numeric.states[interval_index], dtype=np.float64)
        error = _command_minus_current_state(raw, state)
        for phase in PHASES:
            phase_values[phase].append(error[_phase_index(len(raw), phase)])
        targets.append(np.asarray(numeric.actions[interval_index], dtype=np.float64))
        interval_indices.append(interval_index)
        end_commands.append(raw[-1])
        states.append(state)
    return {
        "episode_index": int(manifest["episode_index"]),
        "phase_values": {phase: np.stack(values) for phase, values in phase_values.items()},
        "targets": np.stack(targets),
        "interval_indices": np.asarray(interval_indices),
        "end_commands": np.stack(end_commands),
        "states": np.stack(states),
    }


def _phase_index(command_count: int, phase: float) -> int:
    if command_count < 1 or not 0.0 <= phase <= 1.0:
        raise ValueError("Phase indexing requires command_count >= 1 and phase in [0,1]")
    return round(phase * (command_count - 1))


def _concatenate_gate_episodes(episodes: Mapping[int, Mapping[str, Any]], ids: Sequence[int]) -> dict[str, Any]:
    return {
        "phase_values": {
            phase: np.concatenate([episodes[index]["phase_values"][phase] for index in ids])
            for phase in PHASES
        },
        "targets": np.concatenate([episodes[index]["targets"] for index in ids]),
    }


def _fit_component_phase_affine(data: Mapping[str, Any], *, contract: Any) -> dict[str, Any]:
    target = data["targets"]
    result = {}
    for component in contract.action_components:
        indices = np.asarray(component.indices)
        candidates = []
        for phase in PHASES:
            x = data["phase_values"][phase][:, indices]
            scores = [_dimension_metrics(x[:, offset], target[:, index])["affine_r2"] for offset, index in enumerate(indices)]
            candidates.append((float(np.mean(scores)), -phase, phase))
        _, _, phase = max(candidates)
        x = data["phase_values"][phase]
        dimensions = []
        for index in component.indices:
            metrics = _dimension_metrics(x[:, index], target[:, index])
            dimensions.append({
                "index": index,
                "scale": metrics["affine_scale"],
                "bias": metrics["affine_offset"],
                "fit_r2": metrics["affine_r2"],
            })
        result[component.name] = {"phase": phase, "dimensions": dimensions}
    return result


def _evaluate_frozen_adapter(data: Mapping[str, Any], fitted: Mapping[str, Any], *, contract: Any) -> dict[str, Any]:
    target = data["targets"]
    prediction = np.zeros_like(target)
    for component in contract.action_components:
        spec = fitted[component.name]
        x = data["phase_values"][float(spec["phase"])]
        for dimension in spec["dimensions"]:
            index = int(dimension["index"])
            prediction[:, index] = float(dimension["scale"]) * x[:, index] + float(dimension["bias"])
    return _frozen_prediction_report(prediction, target, contract=contract)


def _frozen_prediction_report(prediction: np.ndarray, target: np.ndarray, *, contract: Any) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Frozen prediction and target must have matching rank-2 shapes")
    components = {}
    for component in contract.action_components:
        indices = np.asarray(component.indices)
        predicted = prediction[:, indices]
        expected = target[:, indices]
        residual = predicted - expected
        centered = expected - expected.mean(axis=0, keepdims=True)
        denominator = float(np.sum(centered**2))
        dimensions = []
        for offset, index in enumerate(indices):
            y = expected[:, offset]
            y_hat = predicted[:, offset]
            target_ss = float(np.sum((y - y.mean()) ** 2))
            dimensions.append({
                "index": int(index),
                "r2": None if target_ss == 0 else float(1.0 - np.sum((y_hat - y) ** 2) / target_ss),
                "mae": float(np.mean(np.abs(y_hat - y))),
                "rmse": float(np.sqrt(np.mean((y_hat - y) ** 2))),
                "correlation": (
                    float(np.corrcoef(y_hat, y)[0, 1])
                    if y_hat.std() > 0 and y.std() > 0 else None
                ),
            })
        components[component.name] = {
            "indices": list(component.indices),
            "count": len(target),
            "r2": None if denominator == 0 else float(1.0 - np.sum(residual**2) / denominator),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "dimensions": dimensions,
        }
    return {"count": len(target), "components": components}


def _gripper_event_report(
    episodes: Mapping[int, Mapping[str, Any]], ids: Sequence[int], *, contract: Any, radius: int = 10
) -> dict[str, Any]:
    reports = {}
    for component in contract.action_components:
        if "gripper" not in component.name:
            continue
        index = component.indices[0]
        aligned = {
            "increasing": {offset: [] for offset in range(-radius, radius + 1)},
            "decreasing": {offset: [] for offset in range(-radius, radius + 1)},
        }
        commands, states, deltas = [], [], []
        switches = 0
        for episode_id in ids:
            episode = episodes[episode_id]
            command = episode["end_commands"][:, index]
            state = episode["states"][:, index]
            target = episode["targets"][:, index]
            commands.append(command)
            states.append(state)
            deltas.append(target)
            switch_threshold = max(1e-6, float(np.ptp(command)) * 1e-6)
            for event in np.flatnonzero(np.abs(np.diff(command)) > switch_threshold) + 1:
                switches += 1
                direction = "increasing" if command[event] > command[event - 1] else "decreasing"
                for offset in range(-radius, radius + 1):
                    position = event + offset
                    if 0 <= position < len(state):
                        aligned[direction][offset].append(
                            (float(state[position]), float(state[position] - state[event]))
                        )
        command_all, state_all, delta_all = map(np.concatenate, (commands, states, deltas))
        reports[component.name] = {
            "switch_count": switches,
            "switch_threshold_rule": "max(1e-6, episode_command_range * 1e-6)",
            "command_unique": np.unique(command_all).tolist()[:64],
            "command_unique_count": len(np.unique(command_all)),
            "command_range": [float(command_all.min()), float(command_all.max())],
            "state_range": [float(state_all.min()), float(state_all.max())],
            "delta_nonzero_fraction": float(np.mean(np.abs(delta_all) > 1e-6)),
            "delta_quantiles": np.quantile(delta_all, [0, 0.01, 0.5, 0.99, 1]).tolist(),
            "event_aligned_state": {
                direction: {
                    str(offset): {
                        "count": len(values),
                        "mean_state": float(np.mean([value[0] for value in values])) if values else None,
                        "mean_change_from_event": (
                            float(np.mean([value[1] for value in values])) if values else None
                        ),
                    }
                    for offset, values in by_offset.items()
                }
                for direction, by_offset in aligned.items()
            },
        }
    return reports


def _print_causal_gate(payload: Mapping[str, Any], *, output_path: Path) -> None:
    print(f"output={output_path}")
    print("=== frozen component-phase affine gate ===")
    for component, spec in payload["fitted_adapter"].items():
        fit = payload["evaluation"]["fit"]["components"][component]
        held = payload["evaluation"]["held_out"]["components"][component]
        print(f"{component:20s} phase={spec['phase']:.2f} fit_r2={fit['r2']!s} "
              f"held_r2={held['r2']!s} held_mae={held['mae']!s} held_rmse={held['rmse']!s}")
    print("=== gripper events ===")
    for component, report in payload["gripper_events"]["held_out"].items():
        print(f"{component:20s} switches={report['switch_count']} unique_commands={report['command_unique_count']} "
              f"nonzero_delta={report['delta_nonzero_fraction']:.6f}")


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
