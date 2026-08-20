import flax.nnx as nnx
import jax
import numpy as np

from openpi.policies import policy as _policy


class _NoisePolicyModel(nnx.Module):
    def __init__(self):
        self.action_horizon = 4
        self.action_dim = 2
        self.max_token_len = 1

    def sample_actions(self, rng, observation, *, noise=None):
        if noise is None:
            noise = jax.random.normal(
                rng,
                (observation.state.shape[0], self.action_horizon, self.action_dim),
            )
        return noise + observation.state[:, None, : self.action_dim]


def _observation() -> dict:
    return {
        "image": {"camera": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"camera": np.ones((), dtype=np.bool_)},
        "state": np.asarray([1.0, 2.0], dtype=np.float32),
    }


def test_reset_restores_jax_sampling_stream():
    policy = _policy.Policy(_NoisePolicyModel(), rng=jax.random.key(7))

    expected = policy.infer(_observation())["actions"]
    policy.infer(_observation())
    policy.reset()

    np.testing.assert_array_equal(policy.infer(_observation())["actions"], expected)


def test_can_return_model_space_actions():
    policy = _policy.Policy(_NoisePolicyModel(), rng=jax.random.key(7))

    result = policy.infer(_observation(), return_model_actions=True)

    np.testing.assert_array_equal(result["model_actions"], result["actions"])


def test_policy_recorder_forwards_async_inference_arguments(tmp_path):
    class _RecordedPolicy:
        def __init__(self):
            self.reset_count = 0

        def infer(self, obs, *, noise=None, return_model_actions=False):
            assert return_model_actions
            return {
                "actions": np.asarray([[1.0]], dtype=np.float32),
                "model_actions": np.asarray([[2.0]], dtype=np.float32),
            }

        def reset(self):
            self.reset_count += 1

    inner = _RecordedPolicy()
    recorder = _policy.PolicyRecorder(inner, str(tmp_path))

    result = recorder.infer({}, return_model_actions=True)
    recorder.reset()

    assert result["model_actions"].item() == 2.0
    assert inner.reset_count == 1
