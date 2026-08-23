"""Held-out prediction evaluation for the frozen upstream WEAVER model.

The data and model boundaries stay deliberately narrow:

* :class:`HeldOutWindowDataset` turns ``BehaviorPrecomputed.window_at`` into a
  deterministic, episode-local evaluation set and rejects the training split.
* :class:`UpstreamWeaverPredictor` calls the frozen model's ``encode_obs``,
  ``generate_latent_rollouts`` and ``decode_obs`` methods.  Future observations
  are explicitly zeroed before rollout so held-out targets cannot leak through.
* :func:`evaluate_prediction_batches` is NumPy-only.  It produces canonical
  23-D state metrics, the last-state persistence baseline, and optional decoded
  RGB metrics without importing PyTorch in CPU-only audit environments.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
import dataclasses
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from openpi.weaver.b1k_dataset_adapter import angular_state_indices

CANONICAL_STATE_REPRESENTATION = "r1pro_compact_state_v1"
ANGULAR_STATE_INDICES = angular_state_indices()


@dataclasses.dataclass(frozen=True)
class WindowRef:
    """One episode-local prediction window."""

    dataset_index: int
    episode_index: int
    anchor_t: int
    first_target_t: int
    last_target_t: int
    manifest_semantic_sha256: str


class HeldOutWindowDataset:
    """Deterministically enumerate valid windows from ``BehaviorPrecomputed``.

    ``BehaviorPrecomputed.__getitem__`` intentionally samples a random anchor
    for training.  Evaluation instead materializes an immutable list of
    ``(episode, anchor_t)`` references and delegates every actual slice to its
    public deterministic ``window_at`` method.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        max_horizon: int,
        stride: int = 1,
        max_windows: int | None = None,
    ) -> None:
        split = getattr(dataset, "split", None)
        if split not in {"val", "test"}:
            raise ValueError(f"Prediction evaluation requires held-out split 'val' or 'test', got {split!r}")
        _positive_int(max_horizon, name="max_horizon")
        _positive_int(stride, name="stride")
        if max_windows is not None:
            _positive_int(max_windows, name="max_windows")
        if int(getattr(dataset, "horizon", -1)) < max_horizon:
            raise ValueError(
                f"BehaviorPrecomputed horizon={getattr(dataset, 'horizon', None)!r} is shorter than "
                f"evaluation max_horizon={max_horizon}"
            )
        n_history = int(getattr(dataset, "n_history", -1))
        _positive_int(n_history, name="dataset.n_history")

        refs: list[WindowRef] = []
        manifests = getattr(dataset, "manifests", None)
        if not isinstance(manifests, Sequence):
            raise TypeError("HeldOutWindowDataset requires BehaviorPrecomputed.manifests")
        for dataset_index, manifest in enumerate(manifests):
            if not isinstance(manifest, Mapping):
                raise TypeError("BehaviorPrecomputed manifests must contain mappings")
            episode_index = int(manifest["episode_index"])
            sampled_count = int(manifest["sampled_count"])
            min_t = n_history - 1
            # Target frames are t+1..t+H, so t+H may equal the final frame.
            max_t = sampled_count - max_horizon - 1
            if max_t < min_t:
                continue
            semantic_sha = str(manifest.get("semantic_sha256", ""))
            for anchor_t in range(min_t, max_t + 1, stride):
                refs.append(
                    WindowRef(
                        dataset_index=dataset_index,
                        episode_index=episode_index,
                        anchor_t=anchor_t,
                        first_target_t=anchor_t + 1,
                        last_target_t=anchor_t + max_horizon,
                        manifest_semantic_sha256=semantic_sha,
                    )
                )
                if max_windows is not None and len(refs) >= max_windows:
                    break
            if max_windows is not None and len(refs) >= max_windows:
                break
        if not refs:
            raise ValueError(
                f"Held-out split {split!r} has no episode-local windows for "
                f"n_history={n_history}, max_horizon={max_horizon}"
            )

        self.dataset = dataset
        self.split = split
        self.n_history = n_history
        self.max_horizon = max_horizon
        self.refs = tuple(refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        sample = self.dataset.window_at(ref.dataset_index, ref.anchor_t, collapse=False)
        _validate_window(sample, n_history=self.n_history, horizon=self.max_horizon)
        result = dict(sample)
        result["eval_provenance"] = {
            "window_index": index,
            "split": self.split,
            "episode_index": ref.episode_index,
            "anchor_t": ref.anchor_t,
            "first_target_t": ref.first_target_t,
            "last_target_t": ref.last_target_t,
            "manifest_semantic_sha256": ref.manifest_semantic_sha256,
        }
        return result


@dataclasses.dataclass(frozen=True)
class PredictionBatch:
    """Canonical predictions and optional decoded images for one horizon."""

    predicted_states: np.ndarray
    target_states: np.ndarray
    persistence_states: np.ndarray
    transition_valid: np.ndarray
    decoded_predicted: Mapping[str, np.ndarray] | None = None
    decoded_target: Mapping[str, np.ndarray] | None = None


class BatchPredictor(Protocol):
    def predict(self, batch: Mapping[str, Any], horizon: int) -> PredictionBatch: ...


class UpstreamWeaverPredictor:
    """Adapter around the frozen upstream WEAVER rollout/decode interfaces."""

    def __init__(
        self,
        model: Any,
        *,
        torch: Any,
        device: str,
        n_history: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        use_ema: bool = True,
        decode_rgb: bool = False,
        decode_chunk_size: int = 16,
    ) -> None:
        _positive_int(n_history, name="n_history")
        if isinstance(decode_chunk_size, bool) or not isinstance(decode_chunk_size, int) or decode_chunk_size < 0:
            raise ValueError("decode_chunk_size must be a non-negative integer")
        mean = np.asarray(state_mean, dtype=np.float32)
        std = np.asarray(state_std, dtype=np.float32)
        if mean.ndim != 1 or std.shape != mean.shape or not np.isfinite(mean).all():
            raise ValueError("state_mean/state_std must be matching finite vectors")
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("state_std must contain finite positive values")
        model_history = getattr(model, "_n_history", n_history)
        if int(model_history) != n_history:
            raise ValueError(f"Model n_history={model_history} does not match evaluator n_history={n_history}")
        for method in ("eval", "encode_obs", "generate_latent_rollouts", "decode_obs"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"Frozen WEAVER model has no callable {method} method")
        if use_ema and not callable(getattr(getattr(model, "ema", None), "use_ema_weights", None)):
            raise TypeError("EMA evaluation requires model.ema.use_ema_weights")

        self.model = model
        self.torch = torch
        self.device = device
        self.n_history = n_history
        self.state_mean = mean
        self.state_std = std
        self.use_ema = bool(use_ema)
        self.decode_rgb = bool(decode_rgb)
        self.decode_chunk_size = decode_chunk_size
        self.model.eval()

    def predict(self, batch: Mapping[str, Any], horizon: int) -> PredictionBatch:
        _positive_int(horizon, name="horizon")
        obs_value = batch.get("obs")
        if not isinstance(obs_value, Mapping) or "states" not in obs_value:
            raise ValueError("Evaluation batch must contain obs.states")
        total = self.n_history + horizon
        obs = {
            key: _to_device(value[:, :total], torch=self.torch, device=self.device) for key, value in obs_value.items()
        }
        actions_value = batch.get("actions")
        if actions_value is None or _shape(actions_value)[1] < total:
            raise ValueError(f"Evaluation actions must contain at least {total} timesteps")
        actions = _to_device(actions_value[:, :total], torch=self.torch, device=self.device)
        memory_value = batch.get("memory")
        memory = None
        if memory_value is not None:
            if not isinstance(memory_value, Mapping):
                raise ValueError("Evaluation memory must be a mapping")
            memory = {
                key: _to_device(value, torch=self.torch, device=self.device) for key, value in memory_value.items()
            }

        transition_mask = target_transition_mask(
            batch.get("transition_valid"),
            n_history=self.n_history,
            horizon=horizon,
        )
        with self.torch.no_grad():
            encoded_target = self.model.encode_obs(obs)
            rollout_input = _zero_future(encoded_target, n_history=self.n_history)
            ema_context = self.model.ema.use_ema_weights() if self.use_ema else nullcontext()
            # Frozen upstream generate_videos applies EMA only around rollout and
            # decodes after restoring online weights.  Preserve that exact rule.
            with ema_context:
                predicted = self.model.generate_latent_rollouts(
                    rollout_input,
                    actions,
                    memory=memory,
                )

            predicted_future = _future_mapping(predicted, self.n_history, horizon)
            target_future = _future_mapping(encoded_target, self.n_history, horizon)
            decoded_predicted = None
            decoded_target = None
            if self.decode_rgb:
                decoded_predicted = _decoded_views(
                    self.model.decode_obs(predicted_future, chunk_size=self.decode_chunk_size)
                )
                decoded_target = _decoded_views(self.model.decode_obs(target_future, chunk_size=self.decode_chunk_size))

        predicted_states = self._canonical(_to_numpy(predicted_future["states"]))
        target_states = self._canonical(_to_numpy(target_future["states"]))
        history_state = _to_numpy(encoded_target["states"][:, self.n_history - 1 : self.n_history])
        persistence_states = self._canonical(np.repeat(history_state, horizon, axis=1))
        return PredictionBatch(
            predicted_states=predicted_states,
            target_states=target_states,
            persistence_states=persistence_states,
            transition_valid=transition_mask,
            decoded_predicted=decoded_predicted,
            decoded_target=decoded_target,
        )

    def _canonical(self, value: np.ndarray) -> np.ndarray:
        if value.shape[-1] != self.state_mean.shape[0]:
            raise ValueError(
                f"Predicted state dim {value.shape[-1]} does not match normalization dim {self.state_mean.shape[0]}"
            )
        return np.asarray(value, dtype=np.float64) * self.state_std + self.state_mean


def target_transition_mask(value: Any, *, n_history: int, horizon: int) -> np.ndarray:
    """Return the real transition mask aligned to future target frames.

    Action/transition at local window position ``n_history - 1`` drives the
    first predicted frame.  Therefore the H prediction masks are not the future
    frame mask slice; they start one element earlier.
    """

    _positive_int(n_history, name="n_history")
    _positive_int(horizon, name="horizon")
    if value is None:
        raise ValueError("Evaluation requires the real transition_valid mask")
    array = _to_numpy(value)
    if array.ndim != 2 or array.dtype != np.bool_:
        raise ValueError(f"transition_valid must have shape (B,T) and bool dtype, got {array.shape}/{array.dtype}")
    start = n_history - 1
    end = start + horizon
    if array.shape[1] < end:
        raise ValueError(f"transition_valid length {array.shape[1]} is shorter than required {end}")
    return np.array(array[:, start:end], dtype=np.bool_, copy=True)


class LpipsMetric(Protocol):
    def __call__(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray: ...


def evaluate_prediction_batches(
    batches: Iterable[Mapping[str, Any]],
    predictor: BatchPredictor,
    *,
    horizons: Sequence[int],
    record_sink: Callable[[Mapping[str, Any]], None] | None = None,
    lpips_metric: LpipsMetric | None = None,
    rgb_data_range: float = 1.0,
) -> dict[str, Any]:
    """Evaluate batches and return a JSON-serializable metric summary."""

    resolved_horizons = _horizons(horizons)
    if not math.isfinite(rgb_data_range) or rgb_data_range <= 0:
        raise ValueError("rgb_data_range must be finite and positive")
    aggregate = _Aggregate(resolved_horizons, rgb_data_range=rgb_data_range)
    observed_windows: set[int | str] = set()
    observed_episodes: set[int | str] = set()

    for batch in batches:
        for horizon in resolved_horizons:
            prediction = predictor.predict(batch, horizon)
            batch_size = _validate_prediction(prediction, horizon=horizon)
            provenance = _provenance_rows(batch.get("eval_provenance"), batch_size=batch_size)
            lpips_values = _compute_lpips(prediction, lpips_metric)
            records = aggregate.add(prediction, horizon=horizon, provenance=provenance, lpips=lpips_values)
            for record in records:
                window_id = record.get("window_index", f"record-{aggregate.record_count}")
                observed_windows.add(window_id)
                observed_episodes.add(record.get("episode_index", "unknown"))
                if record_sink is not None:
                    record_sink(record)

    if aggregate.record_count == 0:
        raise ValueError("Evaluation loader yielded no prediction batches")
    return {
        "schema_version": "b1k_weaver_prediction_eval_v1",
        "state_representation": CANONICAL_STATE_REPRESENTATION,
        "angular_state_indices": list(ANGULAR_STATE_INDICES),
        "horizons": list(resolved_horizons),
        "window_count": len(observed_windows),
        "episode_count": len(observed_episodes),
        "prediction_record_count": aggregate.record_count,
        "metrics": aggregate.summary(),
    }


class DeviceTelemetry:
    """Wall-clock and CUDA peak telemetry with no eager torch import."""

    def __init__(self, torch: Any, device: str) -> None:
        self.torch = torch
        self.device = device
        self._started_at: float | None = None
        self._cuda = False
        self._device_ref: Any = None

    def start(self) -> None:
        if self._started_at is not None:
            raise RuntimeError("Telemetry has already started")
        device_ref = self.torch.device(self.device)
        self._cuda = getattr(device_ref, "type", None) == "cuda"
        self._device_ref = device_ref
        if self._cuda:
            if not self.torch.cuda.is_available():
                raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {self.device}")
            self.torch.cuda.synchronize(device_ref)
            self.torch.cuda.reset_peak_memory_stats(device_ref)
        self._started_at = time.perf_counter()

    def finish(self) -> dict[str, Any]:
        if self._started_at is None:
            raise RuntimeError("Telemetry was not started")
        if self._cuda:
            self.torch.cuda.synchronize(self._device_ref)
        elapsed = time.perf_counter() - self._started_at
        result: dict[str, Any] = {
            "device": str(self.device),
            "cuda": self._cuda,
            "elapsed_seconds": elapsed,
        }
        if self._cuda:
            properties = self.torch.cuda.get_device_properties(self._device_ref)
            result.update(
                {
                    "cuda_device_name": str(properties.name),
                    "cuda_total_memory_bytes": int(properties.total_memory),
                    "cuda_peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated(self._device_ref)),
                    "cuda_peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved(self._device_ref)),
                    "cuda_final_allocated_bytes": int(self.torch.cuda.memory_allocated(self._device_ref)),
                    "cuda_final_reserved_bytes": int(self.torch.cuda.memory_reserved(self._device_ref)),
                }
            )
        return result


class AtomicJsonlWriter:
    """Write JSONL atomically so interrupted evaluations do not look complete."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        self._stream: Any = None

    def __enter__(self) -> AtomicJsonlWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.temporary.open("w", encoding="utf-8", newline="\n")
        return self

    def __call__(self, value: Mapping[str, Any]) -> None:
        if self._stream is None:
            raise RuntimeError("AtomicJsonlWriter is not open")
        self._stream.write(json.dumps(value, sort_keys=True, allow_nan=False))
        self._stream.write("\n")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self._stream is not None:
                if exc_type is None:
                    self._stream.flush()
                    os.fsync(self._stream.fileno())
                self._stream.close()
            if exc_type is None:
                os.replace(self.temporary, self.path)
        finally:
            self.temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@dataclasses.dataclass
class _StateSums:
    absolute: np.ndarray
    squared: np.ndarray
    persistence_absolute: np.ndarray
    persistence_squared: np.ndarray
    elements: np.ndarray
    samples: np.ndarray

    @classmethod
    def create(cls, horizon: int) -> _StateSums:
        def zeros() -> np.ndarray:
            return np.zeros(horizon, dtype=np.float64)

        return cls(zeros(), zeros(), zeros(), zeros(), zeros(), zeros())


@dataclasses.dataclass
class _RgbSums:
    squared: np.ndarray
    pixels: np.ndarray
    samples: np.ndarray
    lpips_sum: np.ndarray
    lpips_samples: np.ndarray

    @classmethod
    def create(cls, horizon: int) -> _RgbSums:
        def zeros() -> np.ndarray:
            return np.zeros(horizon, dtype=np.float64)

        return cls(zeros(), zeros(), zeros(), zeros(), zeros())


class _Aggregate:
    def __init__(self, horizons: tuple[int, ...], *, rgb_data_range: float) -> None:
        self.state = {horizon: _StateSums.create(horizon) for horizon in horizons}
        self.rgb: dict[int, dict[str, _RgbSums]] = {horizon: {} for horizon in horizons}
        self.rgb_data_range = rgb_data_range
        self.record_count = 0

    def add(
        self,
        prediction: PredictionBatch,
        *,
        horizon: int,
        provenance: list[dict[str, Any]],
        lpips: Mapping[str, np.ndarray] | None,
    ) -> list[dict[str, Any]]:
        pred = np.asarray(prediction.predicted_states, dtype=np.float64)
        target = np.asarray(prediction.target_states, dtype=np.float64)
        persistence = np.asarray(prediction.persistence_states, dtype=np.float64)
        valid = np.asarray(prediction.transition_valid, dtype=np.bool_)
        # Periodic torso/arm coordinates use shortest-angle error, matching
        # the state-delta construction contract.  Otherwise an equivalent
        # prediction across +/-pi would incur an artificial 2*pi penalty.
        state_error = _state_error(pred, target)
        persistence_error = _state_error(persistence, target)
        state_sums = self.state[horizon]
        dimension = pred.shape[-1]
        for lead in range(horizon):
            selected = valid[:, lead]
            if not selected.any():
                continue
            state_sums.absolute[lead] += np.abs(state_error[selected, lead]).sum()
            state_sums.squared[lead] += np.square(state_error[selected, lead]).sum()
            state_sums.persistence_absolute[lead] += np.abs(persistence_error[selected, lead]).sum()
            state_sums.persistence_squared[lead] += np.square(persistence_error[selected, lead]).sum()
            state_sums.elements[lead] += int(selected.sum()) * dimension
            state_sums.samples[lead] += int(selected.sum())

        rgb_records: dict[str, list[dict[str, Any]]] = {}
        if prediction.decoded_predicted is not None and prediction.decoded_target is not None:
            for view in sorted(prediction.decoded_predicted):
                pred_rgb = np.asarray(prediction.decoded_predicted[view], dtype=np.float64)
                target_rgb = np.asarray(prediction.decoded_target[view], dtype=np.float64)
                squared = np.square(pred_rgb - target_rgb)
                per_sample_mse = squared.reshape(squared.shape[0], horizon, -1).mean(axis=2)
                rgb_sums = self.rgb[horizon].setdefault(view, _RgbSums.create(horizon))
                pixel_count = int(np.prod(squared.shape[2:]))
                view_lpips = lpips[view] if lpips is not None else None
                for lead in range(horizon):
                    selected = valid[:, lead]
                    if selected.any():
                        rgb_sums.squared[lead] += squared[selected, lead].sum()
                        rgb_sums.pixels[lead] += int(selected.sum()) * pixel_count
                        rgb_sums.samples[lead] += int(selected.sum())
                        if view_lpips is not None:
                            rgb_sums.lpips_sum[lead] += view_lpips[selected, lead].sum()
                            rgb_sums.lpips_samples[lead] += int(selected.sum())
                rgb_records[view] = [
                    {
                        "lead": lead + 1,
                        "valid": bool(valid[index, lead]),
                        "mse": float(per_sample_mse[index, lead]) if valid[index, lead] else None,
                        "psnr_db": _psnr(float(per_sample_mse[index, lead]), self.rgb_data_range)
                        if valid[index, lead]
                        else None,
                        **(
                            {"lpips": float(view_lpips[index, lead]) if valid[index, lead] else None}
                            if view_lpips is not None
                            else {}
                        ),
                    }
                    for index in range(pred.shape[0])
                    for lead in range(horizon)
                ]

        records: list[dict[str, Any]] = []
        for index in range(pred.shape[0]):
            state_rows = []
            for lead in range(horizon):
                is_valid = bool(valid[index, lead])
                error = state_error[index, lead]
                baseline_error = persistence_error[index, lead]
                state_rows.append(
                    {
                        "lead": lead + 1,
                        "transition_valid": is_valid,
                        "mae": float(np.abs(error).mean()) if is_valid else None,
                        "rmse": float(np.sqrt(np.square(error).mean())) if is_valid else None,
                        "persistence_mae": float(np.abs(baseline_error).mean()) if is_valid else None,
                        "persistence_rmse": float(np.sqrt(np.square(baseline_error).mean())) if is_valid else None,
                    }
                )
            record = dict(provenance[index])
            record.update(
                {
                    "schema_version": "b1k_weaver_prediction_sample_v1",
                    "evaluation_horizon": horizon,
                    "state_representation": CANONICAL_STATE_REPRESENTATION,
                    "state_metrics": state_rows,
                }
            )
            if rgb_records:
                record["decoded_rgb_metrics"] = {
                    view: rows[index * horizon : (index + 1) * horizon] for view, rows in rgb_records.items()
                }
            records.append(record)
        self.record_count += len(records)
        return records

    def summary(self) -> dict[str, Any]:
        state_summary = {}
        rgb_summary = {}
        for horizon, sums in self.state.items():
            leads = []
            for lead in range(horizon):
                elements = int(sums.elements[lead])
                leads.append(
                    {
                        "lead": lead + 1,
                        "valid_samples": int(sums.samples[lead]),
                        "mae": _divide(sums.absolute[lead], elements),
                        "rmse": _root_mean(sums.squared[lead], elements),
                        "persistence_mae": _divide(sums.persistence_absolute[lead], elements),
                        "persistence_rmse": _root_mean(sums.persistence_squared[lead], elements),
                    }
                )
            total_elements = int(sums.elements.sum())
            state_summary[str(horizon)] = {
                "by_lead": leads,
                "overall": {
                    "valid_state_elements": total_elements,
                    "mae": _divide(sums.absolute.sum(), total_elements),
                    "rmse": _root_mean(sums.squared.sum(), total_elements),
                    "persistence_mae": _divide(sums.persistence_absolute.sum(), total_elements),
                    "persistence_rmse": _root_mean(sums.persistence_squared.sum(), total_elements),
                },
            }

            if self.rgb[horizon]:
                rgb_summary[str(horizon)] = {}
                for view, rgb_sums in sorted(self.rgb[horizon].items()):
                    view_leads = []
                    for lead in range(horizon):
                        pixels = int(rgb_sums.pixels[lead])
                        mse = _divide(rgb_sums.squared[lead], pixels)
                        row = {
                            "lead": lead + 1,
                            "valid_samples": int(rgb_sums.samples[lead]),
                            "mse": mse,
                            "psnr_db": _psnr(mse, self.rgb_data_range) if mse is not None else None,
                        }
                        if rgb_sums.lpips_samples[lead]:
                            row["lpips"] = _divide(
                                rgb_sums.lpips_sum[lead],
                                int(rgb_sums.lpips_samples[lead]),
                            )
                        view_leads.append(row)
                    total_pixels = int(rgb_sums.pixels.sum())
                    overall_mse = _divide(rgb_sums.squared.sum(), total_pixels)
                    overall = {
                        "valid_pixels": total_pixels,
                        "mse": overall_mse,
                        "psnr_db": _psnr(overall_mse, self.rgb_data_range) if overall_mse is not None else None,
                    }
                    if rgb_sums.lpips_samples.sum():
                        overall["lpips"] = _divide(
                            rgb_sums.lpips_sum.sum(),
                            int(rgb_sums.lpips_samples.sum()),
                        )
                    rgb_summary[str(horizon)][view] = {"by_lead": view_leads, "overall": overall}
        result: dict[str, Any] = {"canonical_state": state_summary}
        if rgb_summary:
            result["decoded_rgb"] = rgb_summary
        return result


def _validate_window(sample: Mapping[str, Any], *, n_history: int, horizon: int) -> None:
    if not isinstance(sample, Mapping):
        raise TypeError("BehaviorPrecomputed.window_at must return a mapping")
    obs = sample.get("obs")
    if not isinstance(obs, Mapping) or "states" not in obs:
        raise ValueError("BehaviorPrecomputed window has no obs.states")
    total = n_history + horizon
    if _shape(obs["states"])[0] != total or _shape(sample.get("actions"))[0] != total:
        raise ValueError(f"BehaviorPrecomputed window must contain exactly {total} timesteps")
    mask = np.asarray(sample.get("transition_valid"))
    if mask.shape != (total,) or mask.dtype != np.bool_:
        raise ValueError(f"BehaviorPrecomputed transition_valid must be bool shape ({total},)")


def _validate_prediction(prediction: PredictionBatch, *, horizon: int) -> int:
    pred = np.asarray(prediction.predicted_states)
    target = np.asarray(prediction.target_states)
    persistence = np.asarray(prediction.persistence_states)
    valid = np.asarray(prediction.transition_valid)
    if pred.ndim != 3 or pred.shape[1] != horizon or pred.shape != target.shape or pred.shape != persistence.shape:
        raise ValueError(
            "Prediction state arrays must share shape (B,H,D), got "
            f"pred={pred.shape}, target={target.shape}, persistence={persistence.shape}"
        )
    if pred.shape[2] != 23:
        raise ValueError(f"Canonical B1K state predictions must be 23-D, got {pred.shape}")
    if valid.shape != pred.shape[:2] or valid.dtype != np.bool_:
        raise ValueError(
            f"Prediction transition_valid must be bool shape {pred.shape[:2]}, got {valid.shape}/{valid.dtype}"
        )
    for name, value in (("predicted_states", pred), ("target_states", target), ("persistence_states", persistence)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    decoded_pred = prediction.decoded_predicted
    decoded_target = prediction.decoded_target
    if (decoded_pred is None) != (decoded_target is None):
        raise ValueError("Decoded prediction and target views must be enabled together")
    if decoded_pred is not None and decoded_target is not None:
        if set(decoded_pred) != set(decoded_target) or not decoded_pred:
            raise ValueError("Decoded prediction and target view names must match and be non-empty")
        for view in decoded_pred:
            pred_rgb = np.asarray(decoded_pred[view])
            target_rgb = np.asarray(decoded_target[view])
            if (
                pred_rgb.ndim != 5
                or pred_rgb.shape[:2] != pred.shape[:2]
                or pred_rgb.shape != target_rgb.shape
                or pred_rgb.shape[2] != 3
            ):
                raise ValueError(f"Decoded view {view!r} must share shape (B,H,3,Y,X), got {pred_rgb.shape}")
            if not np.isfinite(pred_rgb).all() or not np.isfinite(target_rgb).all():
                raise ValueError(f"Decoded view {view!r} contains non-finite values")
    return int(pred.shape[0])


def _compute_lpips(
    prediction: PredictionBatch,
    metric: LpipsMetric | None,
) -> dict[str, np.ndarray] | None:
    if metric is None:
        return None
    if prediction.decoded_predicted is None or prediction.decoded_target is None:
        raise ValueError("LPIPS requires decoded RGB evaluation")
    result = {}
    expected_shape = prediction.transition_valid.shape
    for view in prediction.decoded_predicted:
        value = np.asarray(metric(prediction.decoded_predicted[view], prediction.decoded_target[view]))
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError(f"LPIPS metric for {view!r} must return finite shape {expected_shape}, got {value.shape}")
        result[view] = value.astype(np.float64, copy=False)
    return result


def _provenance_rows(value: Any, *, batch_size: int) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("Evaluation batch is missing eval_provenance")
    rows = [{} for _ in range(batch_size)]
    for key, field in value.items():
        values = _field_values(field, batch_size=batch_size, name=f"eval_provenance.{key}")
        for index, item in enumerate(values):
            rows[index][str(key)] = _json_value(item)
    return rows


def _field_values(value: Any, *, batch_size: int, name: str) -> list[Any]:
    array = _to_numpy(value) if hasattr(value, "shape") or hasattr(value, "detach") else None
    if array is not None:
        if array.ndim == 0:
            if batch_size != 1:
                raise ValueError(f"{name} is scalar for batch_size={batch_size}")
            return [array.item()]
        if array.shape[0] != batch_size:
            raise ValueError(f"{name} length {array.shape[0]} does not match batch_size={batch_size}")
        return [array[index].item() if array[index].ndim == 0 else array[index].tolist() for index in range(batch_size)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != batch_size:
            raise ValueError(f"{name} length {len(value)} does not match batch_size={batch_size}")
        return list(value)
    if batch_size == 1:
        return [value]
    raise ValueError(f"{name} is scalar for batch_size={batch_size}")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _future_mapping(value: Mapping[str, Any], n_history: int, horizon: int) -> dict[str, Any]:
    expected = n_history + horizon
    if "states" not in value:
        raise ValueError("WEAVER latent mapping has no states")
    result = {}
    for key, tensor in value.items():
        if _shape(tensor)[1] < expected:
            raise ValueError(f"WEAVER latent {key!r} is shorter than n_history+horizon={expected}")
        result[key] = tensor[:, n_history:expected]
    return result


def _zero_future(value: Mapping[str, Any], *, n_history: int) -> dict[str, Any]:
    result = {}
    for key, tensor in value.items():
        cloned = tensor.clone() if hasattr(tensor, "clone") else np.array(tensor, copy=True)
        cloned[:, n_history:] = 0
        result[key] = cloned
    return result


def _decoded_views(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {key: _to_numpy(tensor) for key, tensor in value.items() if key != "states"}


def _to_device(value: Any, *, torch: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device=device, non_blocking=True)
    return torch.as_tensor(value, device=device)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _shape(value: Any) -> tuple[int, ...]:
    if value is None or not hasattr(value, "shape"):
        return ()
    return tuple(int(item) for item in value.shape)


def _state_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return canonical state error with periodic coordinates wrapped."""

    if predicted.shape != target.shape or predicted.shape[-1] != 23:
        raise ValueError(f"Canonical state tensors must share final dimension 23, got {predicted.shape}/{target.shape}")
    error = predicted - target
    angular = error[..., ANGULAR_STATE_INDICES]
    pred_angles = predicted[..., ANGULAR_STATE_INDICES]
    target_angles = target[..., ANGULAR_STATE_INDICES]
    canonical = (np.abs(pred_angles) <= np.pi + 1e-6) & (np.abs(target_angles) <= np.pi + 1e-6)
    wrapped = np.arctan2(np.sin(angular), np.cos(angular))
    error[..., ANGULAR_STATE_INDICES] = np.where(canonical, wrapped, angular)
    return error


def _horizons(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("horizons must be a sequence of positive integers")
    result = tuple(sorted(set(values)))
    if not result:
        raise ValueError("At least one evaluation horizon is required")
    for value in result:
        _positive_int(value, name="horizons[]")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _divide(numerator: float | np.floating[Any], denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _root_mean(squared: float | np.floating[Any], elements: int) -> float | None:
    return float(math.sqrt(float(squared) / elements)) if elements else None


def _psnr(mse: float | None, data_range: float) -> float | None:
    # JSON has no portable Infinity value.  ``None`` plus MSE=0 represents a
    # mathematically infinite PSNR without emitting non-standard JSON.
    if mse is None or mse == 0:
        return None
    return float(10.0 * math.log10((data_range * data_range) / mse))
