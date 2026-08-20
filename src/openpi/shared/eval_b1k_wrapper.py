import concurrent.futures
import dataclasses
import logging
import threading
import time

import numpy as np
from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
import torch

from openpi.configs.robots import ROBOT_REGISTRY

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _ActionPlan:
    epoch: int
    request_id: int
    origin_step: int
    processed_actions: np.ndarray
    model_actions: np.ndarray | None
    inference_ms: float


class B1KPolicyWrapper:
    def __init__(
        self,
        policy: BasePolicy,
        robot: str,
        text_prompt: str,
        control_mode: str,
        action_horizon: int,
        max_len: int,
        obs_size: tuple[int, int] = (224, 224),
        async_underflow_policy: str = "error",
    ) -> None:
        # Load robot config from registry
        self.robot = ROBOT_REGISTRY[robot]
        self.policy = policy
        self.text_prompt = text_prompt
        self.control_mode = control_mode
        self.action_horizon = action_horizon
        self.obs_size = obs_size
        self.max_len = max_len
        self.temporal_ensemble_max = 5  # max number of sequences to ensemble

        if self.control_mode == "naive_async" and not 0 < self.action_horizon < self.max_len:
            raise ValueError(
                "naive_async requires 0 < action_horizon < model action horizon; "
                f"got action_horizon={self.action_horizon}, model_horizon={self.max_len}"
            )
        if async_underflow_policy not in {"error", "block"}:
            raise ValueError("async_underflow_policy must be 'error' or 'block'")
        self.async_underflow_policy = async_underflow_policy

        # Extract gripper indices from robot config (end-effectors to preserve)
        self.gripper_indices = []
        for action_config in self.robot.action:
            if action_config.is_eef and action_config.indices is not None:
                self.gripper_indices.extend(action_config.indices)
        # Vectorized action buffers (initialized on first call)
        self.batch_size = None
        self.action_buffer = None  # Shape: (batch, max_sequences, max_horizon, action_dim)
        self.sequence_indices = None  # Shape: (batch, max_sequences) - current position in each sequence
        self.sequence_lengths = None  # Shape: (batch, max_sequences) - total length of each sequence
        self.num_active_sequences = None  # Shape: (batch,) - number of active sequences per batch element
        self.step_counter = None  # Shape: (batch,)

        # Naive async state. All model inference, including the initial plan, runs on this single
        # worker so reset never races a stale GPU call with a new episode.
        self._async_lock = threading.Lock()
        self._async_executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="b1k-naive-async")
            if self.control_mode == "naive_async"
            else None
        )
        self._async_epoch = 0
        self._async_next_action_step = 0
        self._async_next_request_id = 0
        self._async_last_request_origin: int | None = None
        self._async_inflight: concurrent.futures.Future[_ActionPlan] | None = None
        self._async_plan: _ActionPlan | None = None
        self._async_last_delay_steps: int | None = None
        self._async_underflow_count = 0
        self._async_runtime_info: dict = {}

    def reset(self):
        if self.control_mode == "naive_async":
            self._reset_async_state()
        self.batch_size = None
        self.action_buffer = None
        self.sequence_indices = None
        self.sequence_lengths = None
        self.num_active_sequences = None
        self.step_counter = None
        self.policy.reset()

    def close(self) -> None:
        if self._async_executor is None:
            return
        self.reset()
        self._async_executor.shutdown(wait=True, cancel_futures=True)
        self._async_executor = None

    @property
    def runtime_info(self) -> dict:
        return dict(self._async_runtime_info)

    def _reset_async_state(self) -> None:
        with self._async_lock:
            self._async_epoch += 1
            if self._async_inflight is not None:
                # The websocket reset is processed between action requests. Waiting here serializes the
                # old GPU call before Policy.reset() restores the episode RNG.
                try:
                    self._async_inflight.result()
                except Exception:
                    logger.exception("Naive async inference failed while resetting")
            self._async_inflight = None
            self._async_next_action_step = 0
            self._async_next_request_id = 0
            self._async_last_request_origin = None
            self._async_plan = None
            self._async_last_delay_steps = None
            self._async_underflow_count = 0
            self._async_runtime_info = {}

    def _ensure_batch_initialized(self, batch_size: int, action_dim: int = None):
        """Ensure buffers are initialized for the given batch size."""
        if self.batch_size != batch_size or self.action_buffer is None:
            self.batch_size = batch_size

            # For receding_horizon: simple buffer
            # We'll set action_dim when we first see actions
            if action_dim is not None:
                self.action_buffer = np.zeros(
                    (batch_size, self.temporal_ensemble_max, self.max_len, action_dim), dtype=np.float32
                )
                self.sequence_indices = np.zeros((batch_size, self.temporal_ensemble_max), dtype=np.int32)
                self.sequence_lengths = np.zeros((batch_size, self.temporal_ensemble_max), dtype=np.int32)
                self.num_active_sequences = np.zeros(batch_size, dtype=np.int32)

            self.step_counter = np.zeros(batch_size, dtype=np.int32)

    def process_input(self, obs: dict) -> dict:
        """
        Process the input dictionary to match the expected input format for the model.
        """
        prop_state = obs[f"{self.robot.name}::proprio"]
        if prop_state.ndim == 1:
            prop_state = prop_state[None, :]  # Add batch dimension
        batch_size = prop_state.shape[0]
        # Process camera images from robot config
        observations = []
        for camera_key in sorted(self.robot.observations.keys()):
            camera_obs = obs[self.robot.observations[camera_key].obs_key][..., :3]
            if camera_obs.ndim == 3:
                camera_obs = camera_obs[None, ...]  # Add batch dimension
            observations.append(resize_with_pad(camera_obs, *self.obs_size))

        # Pad with zeros if we have fewer than 3 cameras (expected by model)
        while len(observations) < 3:
            observations.append(np.zeros((batch_size, *self.obs_size, 3), dtype=np.uint8))
        img_obs = np.stack(observations, axis=1)  # Shape: (B, 3, H, W, C)
        processed_input = [
            {
                "observation/image_0": img_obs[i, 0],
                "observation/image_1": img_obs[i, 1],
                "observation/image_2": img_obs[i, 2],
                "observation/state": prop_state[i],
                "prompt": self.text_prompt,
            }
            for i in range(batch_size)
        ]
        return processed_input

    def _run_async_inference(
        self,
        policy_input: dict,
        *,
        epoch: int,
        request_id: int,
        origin_step: int,
    ) -> _ActionPlan:
        start_time = time.monotonic()
        result = self.policy.infer(policy_input, return_model_actions=True)
        inference_ms = (time.monotonic() - start_time) * 1000

        processed_actions = np.asarray(result["actions"]).copy()
        if processed_actions.ndim != 2:
            raise ValueError(
                f"Naive async requires an unbatched [horizon, action_dim] chunk, got shape={processed_actions.shape}"
            )
        horizon = min(processed_actions.shape[0], self.max_len)
        if horizon <= self.action_horizon:
            raise ValueError(
                "Naive async needs a model chunk longer than the replanning interval; "
                f"got chunk_horizon={horizon}, action_horizon={self.action_horizon}"
            )
        processed_actions = processed_actions[:horizon]

        model_actions = result.get("model_actions")
        if model_actions is not None:
            model_actions = np.asarray(model_actions).copy()
            if model_actions.ndim != 2 or model_actions.shape[0] < horizon:
                raise ValueError(
                    "model_actions must align with the processed action chunk; "
                    f"got model_shape={model_actions.shape}, processed_shape={processed_actions.shape}"
                )
            model_actions = model_actions[:horizon]

        return _ActionPlan(
            epoch=epoch,
            request_id=request_id,
            origin_step=origin_step,
            processed_actions=processed_actions,
            model_actions=model_actions,
            inference_ms=inference_ms,
        )

    def _submit_async_inference(self, input_obs: dict, origin_step: int) -> None:
        if self._async_executor is None or self._async_inflight is not None:
            raise RuntimeError("Naive async inference submission violated the single-inflight invariant")
        if input_obs[f"{self.robot.name}::proprio"].ndim != 1:
            raise ValueError("naive_async currently supports one evaluator environment per policy server")

        policy_inputs = self.process_input(input_obs)
        if len(policy_inputs) != 1:
            raise ValueError(f"Expected one policy input, got {len(policy_inputs)}")

        request_id = self._async_next_request_id
        self._async_next_request_id += 1
        self._async_last_request_origin = origin_step
        self._async_inflight = self._async_executor.submit(
            self._run_async_inference,
            policy_inputs[0],
            epoch=self._async_epoch,
            request_id=request_id,
            origin_step=origin_step,
        )
        self._async_runtime_info = {
            **self._async_runtime_info,
            "event": "request_submitted",
            "episode_epoch": self._async_epoch,
            "request_id": request_id,
            "origin_step": origin_step,
        }

    def _consume_async_result(self, *, block: bool) -> bool:
        future = self._async_inflight
        if future is None or (not block and not future.done()):
            return False

        plan = future.result()
        self._async_inflight = None
        if plan.epoch != self._async_epoch:
            self._async_runtime_info = {
                **self._async_runtime_info,
                "event": "response_rejected_wrong_epoch",
                "response_epoch": plan.epoch,
                "episode_epoch": self._async_epoch,
            }
            return False

        actual_delay = self._async_next_action_step - plan.origin_step
        if actual_delay < 0 or actual_delay >= len(plan.processed_actions):
            self._async_runtime_info = {
                **self._async_runtime_info,
                "event": "response_rejected_stale",
                "request_id": plan.request_id,
                "origin_step": plan.origin_step,
                "actual_delay_steps": actual_delay,
            }
            return False

        handoff_l2 = None
        if self._async_plan is not None:
            old_index = self._async_next_action_step - self._async_plan.origin_step
            if 0 <= old_index < len(self._async_plan.processed_actions):
                handoff_l2 = float(
                    np.linalg.norm(self._async_plan.processed_actions[old_index] - plan.processed_actions[actual_delay])
                )

        self._async_plan = plan
        self._async_last_delay_steps = actual_delay
        self._async_runtime_info = {
            **self._async_runtime_info,
            "event": "response_accepted",
            "request_id": plan.request_id,
            "origin_step": plan.origin_step,
            "actual_delay_steps": actual_delay,
            "worker_inference_ms": plan.inference_ms,
            "handoff_l2": handoff_l2,
        }
        return True

    def act_naive_async(self, input_obs):
        """Execute an old plan while one new full action chunk is inferred in the background."""
        with self._async_lock:
            current_step = self._async_next_action_step
            self._consume_async_result(block=False)

            # The first observation must synchronously seed a complete plan. Every later inference runs
            # concurrently with execution of the still-valid tail of the current plan.
            if self._async_plan is None:
                if self._async_inflight is None:
                    self._submit_async_inference(input_obs, current_step)
                if not self._consume_async_result(block=True):
                    raise RuntimeError("Failed to initialize the naive async action plan")

            if (
                self._async_inflight is None
                and self._async_last_request_origin is not None
                and current_step - self._async_last_request_origin >= self.action_horizon
            ):
                self._submit_async_inference(input_obs, current_step)
                # Preserve the exact response-vs-action linearization if a very fast worker already finished.
                self._consume_async_result(block=False)

            plan = self._async_plan
            assert plan is not None
            plan_index = current_step - plan.origin_step
            if not 0 <= plan_index < len(plan.processed_actions):
                self._async_underflow_count += 1
                if self.async_underflow_policy == "block" and self._async_inflight is not None:
                    self._consume_async_result(block=True)
                    plan = self._async_plan
                    assert plan is not None
                    plan_index = current_step - plan.origin_step

                if not 0 <= plan_index < len(plan.processed_actions):
                    raise RuntimeError(
                        "Naive async action queue underflow: "
                        f"step={current_step}, plan_origin={plan.origin_step}, "
                        f"plan_horizon={len(plan.processed_actions)}, "
                        f"last_delay={self._async_last_delay_steps}"
                    )

            action = plan.processed_actions[plan_index].copy()
            self._async_runtime_info = {
                **self._async_runtime_info,
                "mode": "naive_async",
                "action_step": current_step,
                "source_request_id": plan.request_id,
                "source_origin_step": plan.origin_step,
                "source_action_index": plan_index,
                "remaining_actions": len(plan.processed_actions) - plan_index - 1,
                "inflight": self._async_inflight is not None,
                "underflow_count": self._async_underflow_count,
            }
            self._async_next_action_step += 1
            return torch.from_numpy(action)

    def _get_current_actions(
        self,
        *,
        batch_range: np.ndarray,
        seq_range: np.ndarray,
        active_mask: np.ndarray,
        current_indices: np.ndarray,
        sequence_lengths: np.ndarray,
    ) -> np.ndarray:
        """Safely gather current actions for active sequences only."""
        safe_lengths = np.maximum(sequence_lengths, 1)
        safe_indices = np.minimum(current_indices, safe_lengths - 1)
        safe_indices = np.where(active_mask, safe_indices, 0)
        batch_idx = batch_range[:, None, None]
        seq_idx = seq_range[None, :, None]
        pos_idx = safe_indices[:, :, None]
        return self.action_buffer[batch_idx, seq_idx, pos_idx].squeeze(-2)

    def act_receding_horizon(self, input_obs):
        """
        receding horizon: execute actions from current plan for up to action_horizon steps, then replan.
        Uses vectorized buffers for efficient batch processing.
        """
        batched = input_obs[f"{self.robot.name}::proprio"].ndim != 1
        input_batch = self.process_input(input_obs)
        batch_size = len(input_batch)
        if self.sequence_indices is None:
            needs_inference = np.ones(batch_size, dtype=bool)
        else:
            needs_inference = (self.sequence_indices[:, 0] >= self.sequence_lengths[:, 0]) | (
                (self.step_counter % self.action_horizon) == 0
            )

        if needs_inference.any():
            indices_needing_inference = np.where(needs_inference)[0]
            # Create sub-batch for elements that need inference
            action = []
            for i in indices_needing_inference:
                action.append(self.policy.infer(input_batch[i]))
            target_action = np.array([a["actions"].copy() for a in action])  # (sub_batch_size, T, action_dim)

            # Initialize buffers on first inference
            if self.action_buffer is None:
                action_dim = target_action.shape[2]
                self._ensure_batch_initialized(batch_size, action_dim)

            # Store actions in buffer
            seq_len = min(target_action.shape[1], self.max_len)
            self.action_buffer[indices_needing_inference, 0, :seq_len] = target_action[:, :seq_len]
            self.sequence_lengths[indices_needing_inference, 0] = seq_len
            self.sequence_indices[indices_needing_inference, 0] = 0

        # Extract next action for each batch element (fully vectorized!)
        batch_range = np.arange(batch_size)
        current_indices = self.sequence_indices[batch_range, 0]
        final_actions = self.action_buffer[batch_range, 0, current_indices]

        # Increment indices (vectorized)
        self.sequence_indices[:, 0] += 1
        self.step_counter += 1
        if not batched:
            final_actions = final_actions[0]
        return torch.from_numpy(final_actions)

    def act_receding_temporal(self, input_obs):
        """
        receding temporal: infer every K steps and smooth actions across recent sequences.
        Uses vectorized buffers for efficient batch processing.
        """
        batched = input_obs[f"{self.robot.name}::proprio"].ndim != 1
        input_batch = self.process_input(input_obs)
        batch_size = len(input_batch)

        # Initialize buffers on first call
        if self.action_buffer is None:
            # Need to infer once to get action_dim
            action = []
            for i in range(batch_size):
                action.append(self.policy.infer(input_batch[i]))
            target_action = np.array([a["actions"].copy() for a in action])  # (B, T, action_dim)
            action_dim = target_action.shape[2]
            self._ensure_batch_initialized(batch_size, action_dim)

            # Store first inference for all batch elements
            seq_len = min(target_action.shape[1], self.max_len)
            self.action_buffer[:, 0, :seq_len] = target_action[:, :seq_len]
            self.sequence_lengths[:, 0] = seq_len
            self.sequence_indices[:, 0] = 0
            self.num_active_sequences[:] = 1

        # Step 1: Run policy for elements that need replanning
        needs_replan = (self.step_counter % self.action_horizon) == 0
        if needs_replan.any():
            indices_needing_replan = np.where(needs_replan)[0]

            # Run inference only on sub-batch
            action = []
            for i in indices_needing_replan:
                action.append(self.policy.infer(input_batch[i]))
            target_action = np.array([a["actions"].copy() for a in action])  # (B, T, action_dim)

            # Add new sequences (vectorized where possible)
            seq_len = min(target_action.shape[1], self.max_len)

            # Handle elements that need shifting
            needs_shift = self.num_active_sequences[indices_needing_replan] >= self.temporal_ensemble_max
            if needs_shift.any():
                shift_indices = indices_needing_replan[needs_shift]
                # Vectorized shift for all elements that need it
                self.action_buffer[shift_indices, :-1] = self.action_buffer[shift_indices, 1:]
                self.sequence_indices[shift_indices, :-1] = self.sequence_indices[shift_indices, 1:]
                self.sequence_lengths[shift_indices, :-1] = self.sequence_lengths[shift_indices, 1:]

            # Calculate insert indices vectorized
            insert_indices = np.where(
                needs_shift, self.temporal_ensemble_max - 1, self.num_active_sequences[indices_needing_replan]
            )

            # Increment active sequences for elements that don't need shifting
            self.num_active_sequences[indices_needing_replan[~needs_shift]] += 1

            # Store new sequences (fully vectorized using advanced indexing)
            self.action_buffer[indices_needing_replan, insert_indices, :seq_len] = target_action[:, :seq_len]
            self.sequence_indices[indices_needing_replan, insert_indices] = 0
            self.sequence_lengths[indices_needing_replan, insert_indices] = seq_len

        # Step 2: Extract and ensemble current actions
        action_dim = self.action_buffer.shape[3]
        max_seq = self.temporal_ensemble_max
        seq_range = np.arange(max_seq)
        batch_range = np.arange(batch_size)
        # Create mask for active sequences (B, max_seq)
        active_mask = seq_range[None, :] < self.num_active_sequences[:, None]
        current_indices = self.sequence_indices[:, :max_seq]  # (B, max_seq)
        actions_current = self._get_current_actions(
            batch_range=batch_range,
            seq_range=seq_range,
            active_mask=active_mask,
            current_indices=current_indices,
            sequence_lengths=self.sequence_lengths[:, :max_seq],
        )  # (B, max_seq, action_dim)
        k = 0.005
        exp_weights = np.exp(k * seq_range)[None, :]  # (1, max_seq)
        masked_weights = exp_weights * active_mask  # (B, max_seq)
        normalized_weights = masked_weights / masked_weights.sum(axis=1, keepdims=True)  # (B, max_seq)
        final_actions = (actions_current * normalized_weights[:, :, None]).sum(axis=1)  # (B, action_dim)

        # Preserve grippers from most recent rollout
        # Get the last active sequence index for each batch element
        last_seq_idx = self.num_active_sequences - 1
        for gripper_idx in self.gripper_indices:
            final_actions[:, gripper_idx] = actions_current[batch_range, last_seq_idx, gripper_idx]

        # Increment sequence indices
        self.sequence_indices[:, :max_seq] += 1

        # Check which sequences are still active (B, max_seq)
        still_active = self.sequence_indices[:, :max_seq] < self.sequence_lengths[:, :max_seq]
        still_active = still_active & active_mask  # Only consider currently active sequences
        # Drop exhausted sequences (compaction per batch element)
        for b in range(batch_size):
            active_in_slot = still_active[b, :]
            if not active_in_slot[: self.num_active_sequences[b]].all():
                # Some sequences exhausted, compact
                active_count = active_in_slot.sum()
                active_indices = np.where(active_in_slot)[0]
                self.action_buffer[b, :active_count] = self.action_buffer[b, active_indices]
                self.sequence_indices[b, :active_count] = self.sequence_indices[b, active_indices]
                self.sequence_lengths[b, :active_count] = self.sequence_lengths[b, active_indices]
                self.num_active_sequences[b] = active_count

        self.step_counter += 1
        if not batched:
            final_actions = final_actions[0]
        return torch.from_numpy(final_actions)

    def act_temporal_ensemble(self, input_obs):
        """
        Temporal ensemble: infer at every step and smooth via exponentially weighted average of all recent action sequences.
        """
        batched = input_obs[f"{self.robot.name}::proprio"].ndim != 1
        input_batch = self.process_input(input_obs)
        batch_size = len(input_batch)
        action = []
        for i in range(batch_size):
            action.append(self.policy.infer(input_batch[i]))
        target_action = np.array([a["actions"].copy() for a in action])  # (B, T, action_dim)
        action_dim = target_action.shape[2]

        # Initialize buffers on first call
        if self.action_buffer is None:
            self._ensure_batch_initialized(batch_size, action_dim)
            self.num_active_sequences[:] = 0

        # Vectorized shift for all elements that need it, increment active counts otherwise
        needs_shift = self.num_active_sequences >= self.temporal_ensemble_max
        if needs_shift.any():
            shift_indices = np.where(needs_shift)[0]
            self.action_buffer[shift_indices, :-1] = self.action_buffer[shift_indices, 1:]
            self.sequence_indices[shift_indices, :-1] = self.sequence_indices[shift_indices, 1:]
            self.sequence_lengths[shift_indices, :-1] = self.sequence_lengths[shift_indices, 1:]
        self.num_active_sequences[~needs_shift] += 1

        # Insert new sequences to action buffer
        insert_indices = np.where(needs_shift, self.temporal_ensemble_max - 1, self.num_active_sequences - 1)
        seq_len = min(target_action.shape[1], self.max_len)
        batch_range = np.arange(batch_size)
        self.action_buffer[batch_range, insert_indices, :seq_len] = target_action[:, :seq_len]
        self.sequence_indices[batch_range, insert_indices] = 0
        self.sequence_lengths[batch_range, insert_indices] = seq_len

        # Extract current actions
        seq_range = np.arange(self.temporal_ensemble_max)
        active_mask = seq_range[None, :] < self.num_active_sequences[:, None]
        current_indices = self.sequence_indices[:, : self.temporal_ensemble_max]  # (B, max_seq)
        actions_current = self._get_current_actions(
            batch_range=batch_range,
            seq_range=seq_range,
            active_mask=active_mask,
            current_indices=current_indices,
            sequence_lengths=self.sequence_lengths[:, : self.temporal_ensemble_max],
        )  # (B, max_seq, action_dim)

        k = 0.005
        exp_weights = np.exp(k * seq_range)[None, :]  # (1, max_seq)
        masked_weights = exp_weights * active_mask  # (B, max_seq)
        normalized_weights = masked_weights / masked_weights.sum(axis=1, keepdims=True)  # (B, max_seq)
        final_actions = (actions_current * normalized_weights[:, :, None]).sum(axis=1)  # (B, action_dim)

        # Preserve grippers from most recent rollout
        for gripper_idx in self.gripper_indices:
            final_actions[:, gripper_idx] = target_action[:, 0, gripper_idx]

        # Increment all sequence indices
        self.sequence_indices[:, : self.temporal_ensemble_max] += 1

        # Drop exhausted sequences (compaction per batch element)
        still_active = (
            self.sequence_indices[:, : self.temporal_ensemble_max]
            < self.sequence_lengths[:, : self.temporal_ensemble_max]
        )
        still_active = still_active & active_mask
        for b in range(batch_size):
            active_in_slot = still_active[b, :]
            if not active_in_slot[: self.num_active_sequences[b]].all():
                active_count = active_in_slot.sum()
                active_indices = np.where(active_in_slot)[0]
                self.action_buffer[b, :active_count] = self.action_buffer[b, active_indices]
                self.sequence_indices[b, :active_count] = self.sequence_indices[b, active_indices]
                self.sequence_lengths[b, :active_count] = self.sequence_lengths[b, active_indices]
                self.num_active_sequences[b] = active_count

        if not batched:
            final_actions = final_actions[0]
        return torch.from_numpy(final_actions)

    def act(self, input_obs):
        """Dispatch to the appropriate action method based on control mode."""
        if self.control_mode == "receding_temporal":
            return self.act_receding_temporal(input_obs)
        if self.control_mode == "receding_horizon":
            return self.act_receding_horizon(input_obs)
        if self.control_mode == "naive_async":
            return self.act_naive_async(input_obs)
        if self.control_mode == "temporal_ensemble":
            return self.act_temporal_ensemble(input_obs)
        raise ValueError(f"Unknown control mode: {self.control_mode}")
