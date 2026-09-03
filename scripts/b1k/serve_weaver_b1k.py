# ruff: noqa: SLF001
"""Serve full WEAVER-guided BEHAVIOR-1K steering over the official WebSocket.

This launcher is intentionally separate from ``serve_b1k.py``.  It keeps the
official OpenPI input transform and WebSocket protocol, while selecting one
candidate from an immutable policy pool with the frozen WEAVER WM+RM+critic.
The online path uses only evaluator observations; future simulator state and
oracle rewards are never passed to the selector.
"""

from __future__ import annotations

from collections import deque
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from openpi.weaver import b1k_contract
from openpi.weaver import candidate_pool
from openpi.weaver import causal_action_adapter
from openpi.weaver import hybrid_action
from openpi.weaver import steering_runtime
from openpi.weaver import upstream_config
from openpi.weaver import upstream_runtime

LOGGER = logging.getLogger(__name__)
TASK_PROMPT_CATALOG_SCHEMA = "behavior_task_prompts_v1"


def _canonical_prompt_catalog_sha256(prompts: dict[str, str]) -> str:
    payload = json.dumps(
        {name: prompts[name] for name in sorted(prompts)},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_task_prompt_catalog(dataset_dir: str | Path) -> dict[str, str] | None:
    """Load the prompt catalog emitted alongside a prepared B1K dataset.

    A present catalog is authoritative and therefore fails closed when its
    schema, prompt entries, or dataset provenance are invalid.  A missing
    catalog is allowed for backwards compatibility with old artifacts; the
    caller may then use the small built-in registry as a fallback.
    """

    root = Path(dataset_dir).expanduser().resolve()
    catalog_path = root / "task_prompts.json"
    if not catalog_path.is_file():
        return None
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read task prompt catalog {catalog_path}: {exc}") from exc
    if not isinstance(catalog, dict) or catalog.get("schema_version") != TASK_PROMPT_CATALOG_SCHEMA:
        raise ValueError(
            f"Unsupported task prompt catalog schema in {catalog_path}: "
            f"{catalog.get('schema_version') if isinstance(catalog, dict) else None!r}"
        )

    dataset_hash = catalog.get("dataset_metadata_sha256")
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise ValueError(f"Task prompt catalog has invalid dataset_metadata_sha256: {catalog_path}")
    try:
        int(dataset_hash, 16)
    except ValueError as exc:
        raise ValueError(f"Task prompt catalog has invalid dataset_metadata_sha256: {catalog_path}") from exc

    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"Task prompt catalog requires sibling summary.json: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read dataset summary {summary_path}: {exc}") from exc
    summary_hash = summary.get("dataset_metadata_sha256") if isinstance(summary, dict) else None
    if summary_hash != dataset_hash:
        raise ValueError(
            "Task prompt catalog dataset hash does not match summary.json: "
            f"catalog={dataset_hash!r}, summary={summary_hash!r}"
        )

    prompts = catalog.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError(f"Task prompt catalog has no prompts mapping: {catalog_path}")
    normalized: dict[str, str] = {}
    for raw_name, raw_prompt in prompts.items():
        name = str(raw_name).strip()
        if name.startswith("b1k/"):
            name = name.split("/", 1)[1].strip()
        prompt = raw_prompt.strip() if isinstance(raw_prompt, str) else ""
        if not name or "/" in name or not prompt:
            raise ValueError(f"Invalid task prompt catalog entry {raw_name!r}: {catalog_path}")
        if name in normalized and normalized[name] != prompt:
            raise ValueError(f"Conflicting prompt catalog entries for task {name!r}: {catalog_path}")
        normalized[name] = prompt
    catalog_hash = catalog.get("task_catalog_sha256")
    if not isinstance(catalog_hash, str) or len(catalog_hash) != 64:
        raise ValueError(f"Task prompt catalog has invalid task_catalog_sha256: {catalog_path}")
    try:
        int(catalog_hash, 16)
    except ValueError as exc:
        raise ValueError(f"Task prompt catalog has invalid task_catalog_sha256: {catalog_path}") from exc
    if catalog_hash != _canonical_prompt_catalog_sha256(normalized):
        raise ValueError(f"Task prompt catalog task_catalog_sha256 does not match prompts: {catalog_path}")
    provenance = summary.get("dataset_provenance") if isinstance(summary, dict) else None
    summary_catalog_hash = provenance.get("task_catalog_sha256") if isinstance(provenance, dict) else None
    legacy_catalog = summary.get("task_catalog") if isinstance(summary, dict) else None
    if summary_catalog_hash is None and isinstance(legacy_catalog, dict):
        summary_catalog_hash = legacy_catalog.get("sha256")
    if summary_catalog_hash != catalog_hash:
        raise ValueError(
            "Task prompt catalog task hash does not match summary.json: "
            f"catalog={catalog_hash!r}, summary={summary_catalog_hash!r}"
        )
    return normalized


def _resolve_task(
    task: str,
    prompt_override: str | None,
    registry: dict[str, dict[str, str]],
    prompt_catalog: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    task_value = str(task).strip()
    if not task_value:
        raise ValueError("--task must be a non-empty task name or 'bucket/name'")
    if "/" in task_value:
        task_bucket, task_name = task_value.split("/", 1)
    else:
        task_bucket, task_name = "b1k", task_value
    if not task_bucket or not task_name:
        raise ValueError(f"Invalid --task {task!r}; expected task name or 'bucket/name'")
    if prompt_override is not None:
        prompt = prompt_override.strip()
        if not prompt:
            raise ValueError("--prompt must be non-empty when supplied")
        return task_bucket, task_name, prompt
    if task_bucket == "b1k" and prompt_catalog is not None:
        prompt = prompt_catalog.get(task_name)
        if prompt is not None:
            return task_bucket, task_name, prompt
    try:
        return task_bucket, task_name, registry[task_bucket][task_name]
    except KeyError as exc:
        available = sorted(registry.get(task_bucket, {}))
        raise ValueError(
            f"No registered prompt for task {task_bucket}/{task_name}. "
            "Pass --prompt with the exact BDDL/OpenPI instruction. "
            f"Known local tasks for {task_bucket!r}: {available}"
        ) from exc


def _require_checkpoint_policy_contract(policy_cfg: Any, *, robot: str) -> Any:
    """Keep checkpoint dataset/assets immutable while validating embodiment."""

    configured_robot = getattr(policy_cfg.data, "robot_config_name", None)
    if configured_robot != robot:
        raise ValueError(
            "The borrowed OpenPI checkpoint must keep its training data contract; "
            f"config robot={configured_robot!r}, requested robot={robot!r}"
        )
    return policy_cfg


@dataclasses.dataclass
class Args:
    robot: str
    task: str
    policy_config: str
    policy_dir: str
    weaver_config: str
    weaver_checkpoint: str
    task_encoder_model: str
    hybrid_action_contract: str
    norm_stats: str
    dataset_dir: str
    prompt: str | None = None
    port: int = 8000
    num_candidates: int = 4
    control_hz: float = 30.0
    target_hz: float = 5.0
    imagination_horizon: int = 5
    num_flow_steps: int = 10
    action_horizon: int = 6
    criterion: str = "advantage"
    checkpoint_mode: Literal["resume", "identity", "b1k_23d_fresh"] = "resume"
    resize_mode: Literal["letterbox", "center-crop", "stretch"] = "letterbox"
    device: str = "cuda"
    use_ema: bool = True


def _load_norm(
    path: str | Path, *, expected_hybrid_contract: hybrid_action.HybridActionContract
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = yaml.safe_load(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("source_split") != "train":
        raise ValueError("WEAVER norm stats must be a train-split JSON artifact")
    stats = payload.get("norm_stats")
    if not isinstance(stats, dict):
        raise ValueError("norm_stats artifact has no norm_stats mapping")
    provenance = payload.get("hybrid_action_contract")
    if not isinstance(provenance, dict) or provenance.get("source_audit_sha256") != (
        expected_hybrid_contract.source_audit_sha256
    ):
        raise ValueError("WEAVER norm stats were not computed from the selected hybrid action contract")

    def pair(name: str, representation: str) -> tuple[np.ndarray, np.ndarray]:
        value = stats.get(name)
        if not isinstance(value, dict) or value.get("representation") != representation:
            raise ValueError(f"Invalid {name} normalization representation")
        mean = np.asarray(value.get("mean"), dtype=np.float32)
        std = np.asarray(value.get("std"), dtype=np.float32)
        if mean.shape != (23,) or std.shape != (23,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError(f"{name} normalization must contain finite 23-D mean/std")
        if np.any(std <= 0):
            raise ValueError(f"{name} normalization std must be positive")
        return mean, std

    state_mean, state_std = pair("state", "r1pro_compact_state_v1")
    action_mean, action_std = pair("actions", hybrid_action.HYBRID_REPRESENTATION)
    return state_mean, state_std, action_mean, action_std


def _resize_batch(batch: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    """Match the offline latent-cache resize contract byte-for-byte."""
    target_h, target_w = size
    source_h, source_w = batch.shape[-2:]
    if mode == "stretch":
        return torch.nn.functional.interpolate(batch, size=size, mode="bilinear", align_corners=False)
    scale = min(target_h / source_h, target_w / source_w) if mode == "letterbox" else max(
        target_h / source_h, target_w / source_w
    )
    raw_h = max(1, round(source_h * scale))
    raw_w = max(1, round(source_w * scale))
    resized_h, resized_w = (
        (min(target_h, raw_h), min(target_w, raw_w))
        if mode == "letterbox"
        else (max(target_h, raw_h), max(target_w, raw_w))
    )
    resized = torch.nn.functional.interpolate(
        batch, size=(resized_h, resized_w), mode="bilinear", align_corners=False
    )
    if mode == "letterbox":
        pad_h, pad_w = target_h - resized_h, target_w - resized_w
        return torch.nn.functional.pad(
            resized,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            value=0.0,
        )
    top, left = (resized_h - target_h) // 2, (resized_w - target_w) // 2
    result = resized[:, :, top : top + target_h, left : left + target_w]
    if result.shape[-2:] != (target_h, target_w):
        raise RuntimeError(f"center-crop produced wrong shape: {tuple(result.shape)}")
    return result


class _RawCandidatePolicy:
    """Bridge raw evaluator observations to the existing OpenPI infer API."""

    def __init__(self, base_policy: Any, processor: Any) -> None:
        self.base_policy = base_policy
        self.processor = processor

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        processed = self.processor.process_input(observation)
        if len(processed) != 1:
            raise ValueError("WEAVER WebSocket serving supports one unbatched evaluator stream")
        return self.base_policy.infer(processed[0], noise=noise)


class OnlineWeaverPolicy:
    """Stateful 5 Hz context/history buffer around the same-pool selector."""

    def __init__(self, *, base_policy: Any, processor: Any, model: Any,
                 hybrid_contract: hybrid_action.HybridActionContract, state_mean: np.ndarray,
                 state_std: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray,
                 task_prompt: str, contract: b1k_contract.B1KWeaverContract, args: Args) -> None:
        self.base_policy = base_policy
        self.processor = processor
        self.model = model
        self.hybrid_contract = hybrid_contract
        self.state_mean, self.state_std = state_mean, state_std
        self.action_mean, self.action_std = action_mean, action_std
        self.task_prompt = task_prompt
        self.contract = contract
        self.args = args
        self.raw_policy = _RawCandidatePolicy(base_policy, processor)
        self._frames: deque[dict[str, torch.Tensor]] = deque(maxlen=int(model._n_history))
        memory_capacity = max(1, int(getattr(model, "_n_memory_frames", 0)) * int(getattr(model, "_t_memory", 1)) + 1)
        self._real_frames: deque[dict[str, torch.Tensor]] = deque(maxlen=memory_capacity)
        self._history: deque[torch.Tensor] = deque(maxlen=int(model._n_history))
        self._remaining: np.ndarray | None = None
        self._remaining_latent: np.ndarray | None = None
        ratio = args.control_hz / args.target_hz
        if not np.isfinite(ratio) or not float(ratio).is_integer():
            raise ValueError("control_hz / target_hz must be a positive integer for online interval commits")
        self._control_steps_per_interval = int(ratio)
        self._actions_since_commit = 0
        self._reset_stream()

    def reset(self) -> None:
        self.base_policy.reset()
        self._reset_stream()

    def _reset_stream(self) -> None:
        self._frames.clear()
        self._real_frames.clear()
        self._history.clear()
        self._remaining = None
        self._remaining_latent = None
        self._actions_since_commit = 0

    def _encode_frame(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        raw_state = np.asarray(observation["robot_r1::proprio"])
        state = self.contract.extract_state(raw_state).astype(np.float32)
        state_norm = (state - self.state_mean) / self.state_std
        views = self.contract.extract_online_views(observation)
        result: dict[str, torch.Tensor] = {}
        image_size = tuple(upstream_config.architecture_contract(self.model_config).image_size)
        with torch.no_grad():
            for key in self.model._img_keys:
                if key not in views:
                    raise KeyError(f"WEAVER view {key!r} is missing from the R1Pro observation")
                image = np.asarray(views[key])[..., :3]
                tensor = torch.as_tensor(image, dtype=torch.float32, device=self.args.device).permute(2, 0, 1)
                # Match the pinned upstream real-frame path exactly: the
                # frozen SD3 encoder receives [0,1] after the resize, while
                # its internal preprocessing handles the latent projection.
                tensor = tensor.div(255.0)
                tensor = _resize_batch(tensor[None], image_size, self.args.resize_mode)
                features = self.model.im_encoder(tensor)
                if features.ndim != 3 or features.shape[0] != 1:
                    raise ValueError(f"WEAVER image encoder must return (1,P,D), got {tuple(features.shape)}")
                result[key] = features.unsqueeze(1)
        result["states"] = torch.as_tensor(state_norm, dtype=torch.float32, device=self.args.device).view(1, 1, -1)
        return result

    @property
    def model_config(self) -> dict[str, Any]:
        return self._model_config

    def set_model_config(self, value: dict[str, Any]) -> None:
        self._model_config = value

    def _commit_frame(self, frame: dict[str, torch.Tensor], latent_action: np.ndarray | None) -> None:
        self._real_frames.append(frame)
        self._frames.append(frame)
        if latent_action is None:
            self._history.append(torch.zeros(1, 1, 23, device=self.args.device))
        else:
            self._history.append(torch.as_tensor(latent_action[None, None], dtype=torch.float32, device=self.args.device))

    def _context(self) -> tuple[dict[str, torch.Tensor], torch.Tensor, Any]:
        if not self._frames:
            raise RuntimeError("WEAVER context has not received an evaluator frame")
        while len(self._frames) < int(self.model._n_history):
            self._frames.appendleft(self._frames[0])
        while len(self._history) < int(self.model._n_history):
            self._history.appendleft(torch.zeros(1, 1, 23, device=self.args.device))
        context = {key: torch.cat([frame[key] for frame in self._frames], dim=1) for key in self.model._img_keys}
        context["states"] = torch.cat([frame["states"] for frame in self._frames], dim=1)
        history = torch.cat(list(self._history), dim=1)
        memory = None
        if getattr(self.model, "_use_memory", False) and int(self.model._n_memory_frames) > 0:
            frames = self._real_frames or list(self._frames)
            head = len(frames)
            selected = [frames[max(0, head - (int(self.model._n_memory_frames) - i) * int(self.model._t_memory))]
                       for i in range(int(self.model._n_memory_frames))]
            memory_obs = {key + "_features": torch.cat([frame[key] for frame in selected], dim=1) for key in self.model._img_keys}
            memory_obs["states"] = torch.cat([frame["states"] for frame in selected], dim=1)
            memory = self.model.encode_memory_obs(memory_obs)
        return context, history, memory

    def act(self, observation: dict[str, Any]) -> torch.Tensor:
        if np.asarray(observation["robot_r1::proprio"]).ndim != 1:
            raise ValueError("WEAVER serving accepts one unbatched R1Pro stream")
        frame = self._encode_frame(observation)
        if not self._frames:
            self._commit_frame(frame, None)
        elif self._actions_since_commit == self._control_steps_per_interval:
            if self._remaining_latent is None or len(self._remaining_latent) == 0:
                raise RuntimeError("Executed a complete control interval without a selected WEAVER action")
            self._commit_frame(frame, self._remaining_latent[0])
            self._remaining_latent = self._remaining_latent[1:]
            self._actions_since_commit = 0
        if self._remaining is None or len(self._remaining) == 0:
            pool = candidate_pool.sample_policy_candidates(self.raw_policy, observation,
                                                            num_candidates=self.args.num_candidates, action_dim=23)
            current_state = self.contract.extract_state(np.asarray(observation["robot_r1::proprio"]))
            edges = np.arange(pool.actions.shape[1] + 1, dtype=np.float64) / self.args.control_hz
            adapted = causal_action_adapter.adapt_candidate_pool_hybrid(
                pool, self.hybrid_contract, current_state, edges,
                weaver_action_mean=self.action_mean, weaver_action_std=self.action_std,
                target_hz=self.args.target_hz,
                max_horizon=self.args.imagination_horizon,
            )
            context, history, memory = self._context()
            scored = steering_runtime.imagine_and_score(
                self.model, context=context, history_actions=history, adapted_actions=adapted,
                task={"text": self.task_prompt}, n_history=int(self.model._n_history), candidate_pool=pool,
                device=self.args.device, gamma=float(getattr(self.model, "_gamma", 0.995)),
                criterion=self.args.criterion, memory_tokens=memory,
            )
            selected = int(scored.best_index)
            execute_steps = self.args.imagination_horizon * self._control_steps_per_interval
            if execute_steps > pool.actions.shape[1]:
                raise ValueError("Selected policy chunk is shorter than its scored execution window")
            self._remaining = np.array(pool.actions[selected, :execute_steps], copy=True)
            self._remaining_latent = np.array(adapted.normalized_weaver_actions[selected], copy=True)
            LOGGER.info(
                "WEAVER replan pool=%s selected=%d candidate=%s criterion=%s scores=%s execute_steps=%d",
                pool.actions_sha256,
                selected,
                pool.candidate_ids[selected],
                scored.criterion,
                np.asarray(scored.scores).tolist(),
                execute_steps,
            )
        action = np.array(self._remaining[0], copy=True)
        self._remaining = self._remaining[1:]
        self._actions_since_commit += 1
        return torch.as_tensor(action, dtype=torch.float32, device="cpu")


def main(args: Args) -> None:
    from openpi.configs.tasks import TASK_REGISTRY  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.serving import websocket_b1k_server  # noqa: PLC0415
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper  # noqa: PLC0415

    required_paths = {
        "policy_dir": args.policy_dir,
        "weaver_config": args.weaver_config,
        "weaver_checkpoint": args.weaver_checkpoint,
        "task_encoder_model": args.task_encoder_model,
        "hybrid_action_contract": args.hybrid_action_contract,
        "norm_stats": args.norm_stats,
        "dataset_dir": args.dataset_dir,
    }
    empty_paths = tuple(name for name, value in required_paths.items() if not str(value).strip())
    if empty_paths:
        raise ValueError(f"Required WEAVER serving paths are empty: {empty_paths}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("WEAVER serving requested CUDA but torch.cuda.is_available() is false")
    prompt_catalog = _load_task_prompt_catalog(args.dataset_dir)
    task_bucket, task_name, prompt = _resolve_task(
        args.task,
        args.prompt,
        TASK_REGISTRY,
        prompt_catalog,
    )
    raw_config = yaml.safe_load(Path(args.weaver_config).expanduser().resolve().read_text(encoding="utf-8"))
    config, _ = upstream_config.build_behavior_config(raw_config, dataset_root=args.dataset_dir,
                                                       norm_stats_path=args.norm_stats, dynamics_only=False)
    if args.criterion not in {"reward", "advantage"}:
        raise ValueError("criterion must be 'reward' or 'advantage'")
    model = upstream_runtime.build_model(
        config,
        device=args.device,
        task_encoder_model=args.task_encoder_model,
    )
    upstream_runtime.load_checkpoint_into_model(model, args.weaver_checkpoint,
                                                mode=args.checkpoint_mode,
                                                device=args.device)
    if args.use_ema:
        model.ema.ema.copy_to(parameter for parameter in model.parameters() if parameter.requires_grad)
    model.eval()
    hybrid_contract = hybrid_action.load_audited_contract(args.hybrid_action_contract)
    state_mean, state_std, action_mean, action_std = _load_norm(
        args.norm_stats, expected_hybrid_contract=hybrid_contract
    )

    policy_cfg = __import__("openpi.training.config", fromlist=["get_config"]).get_config(args.policy_config)
    policy_cfg = _require_checkpoint_policy_contract(policy_cfg, robot=args.robot)
    base = policy_config.create_trained_policy(policy_cfg, args.policy_dir,
                                               sample_kwargs={"num_steps": args.num_flow_steps}, default_prompt=prompt)
    causal_action_adapter.require_world_model_horizon(
        int(policy_cfg.model.action_horizon), args.imagination_horizon,
        control_hz=args.control_hz, target_hz=args.target_hz,
    )
    model._gamma = float(config.get("model", {}).get("discount_factor", 0.995))
    processor = B1KPolicyWrapper(policy=base, robot=args.robot, text_prompt=prompt,
                                 control_mode="receding_horizon", action_horizon=args.action_horizon,
                                 max_len=int(policy_cfg.model.action_horizon))
    policy = OnlineWeaverPolicy(base_policy=base, processor=processor, model=model,
                                hybrid_contract=hybrid_contract, state_mean=state_mean, state_std=state_std,
                                action_mean=action_mean, action_std=action_std, task_prompt=prompt,
                                contract=b1k_contract.build_r1pro_contract(), args=args)
    policy.set_model_config(config)
    metadata = getattr(base, "metadata", {})
    websocket_b1k_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port,
                                               metadata={**metadata, "weaver_steering": True}).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(__import__("tyro").cli(Args))
