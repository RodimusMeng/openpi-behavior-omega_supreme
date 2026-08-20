import threading

import numpy as np
import pytest

from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper


class _ControlledChunkPolicy:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.calls = 0
        self.release_second_call = threading.Event()
        self.reset_count = 0

    def infer(self, obs, *, noise=None, return_model_actions=False):
        call = self.calls
        self.calls += 1
        if call == 1 and not self.release_second_call.wait(timeout=5):
            raise TimeoutError("Test did not release the second inference")

        values = call * 100 + np.arange(self.horizon, dtype=np.float32)
        actions = np.repeat(values[:, None], 2, axis=1)
        result = {"actions": actions}
        if return_model_actions:
            result["model_actions"] = np.pad(actions, ((0, 0), (0, 1)))
        return result

    def reset(self):
        self.reset_count += 1


def _make_wrapper(policy, *, horizon: int, replan_interval: int, underflow: str = "error"):
    wrapper = B1KPolicyWrapper(
        policy=policy,
        robot="b1k/R1Pro",
        text_prompt="test",
        control_mode="naive_async",
        action_horizon=replan_interval,
        max_len=horizon,
        async_underflow_policy=underflow,
    )
    # These tests exercise scheduling and queue alignment, not image preprocessing.
    wrapper.process_input = lambda obs: [obs]
    return wrapper


def _obs(wrapper, plan_id: int) -> dict:
    return {
        f"{wrapper.robot.name}::proprio": np.zeros(3, dtype=np.float32),
        "plan_id": plan_id,
    }


def test_naive_async_skips_actions_executed_during_inference():
    policy = _ControlledChunkPolicy(horizon=8)
    wrapper = _make_wrapper(policy, horizon=8, replan_interval=3)
    try:
        assert wrapper.act(_obs(wrapper, 0))[0].item() == 0
        assert wrapper.act(_obs(wrapper, 0))[0].item() == 1
        assert wrapper.act(_obs(wrapper, 0))[0].item() == 2

        # Request 1 originates at global step 3. Old-plan actions continue while it is blocked.
        assert wrapper.act(_obs(wrapper, 1))[0].item() == 3
        assert wrapper.act(_obs(wrapper, 1))[0].item() == 4

        policy.release_second_call.set()
        wrapper._async_inflight.result(timeout=5)  # noqa: SLF001

        # At step 5, actual_d = 5 - 3 = 2, so the first new action is chunk[2].
        assert wrapper.act(_obs(wrapper, 1))[0].item() == 102
        assert wrapper.runtime_info["actual_delay_steps"] == 2
        assert wrapper.runtime_info["source_origin_step"] == 3
        assert wrapper.runtime_info["source_action_index"] == 2
    finally:
        policy.release_second_call.set()
        wrapper.close()


def test_naive_async_reports_queue_underflow_instead_of_holding_last_action():
    policy = _ControlledChunkPolicy(horizon=5)
    wrapper = _make_wrapper(policy, horizon=5, replan_interval=3)
    try:
        for expected in range(5):
            assert wrapper.act(_obs(wrapper, int(expected >= 3)))[0].item() == expected

        with pytest.raises(RuntimeError, match="queue underflow"):
            wrapper.act(_obs(wrapper, 1))
    finally:
        policy.release_second_call.set()
        wrapper.close()


def test_naive_async_reset_discards_plan_and_restarts_global_step():
    policy = _ControlledChunkPolicy(horizon=8)
    policy.release_second_call.set()
    wrapper = _make_wrapper(policy, horizon=8, replan_interval=3)
    try:
        assert wrapper.act(_obs(wrapper, 0))[0].item() == 0
        wrapper.reset()

        # The policy's next synthetic chunk starts at 100, but its action index is reset to zero.
        assert wrapper.act(_obs(wrapper, 1))[0].item() == 100
        assert wrapper.runtime_info["action_step"] == 0
        assert wrapper.runtime_info["source_action_index"] == 0
        assert policy.reset_count == 1
    finally:
        wrapper.close()
