from __future__ import annotations

from collections.abc import Mapping

import pytest

torch = pytest.importorskip("torch")

from openpi.weaver import training_runtime  # noqa: E402


class _TinyFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, xt, actions, timestep, memory=None):
        del actions, timestep, memory
        return {key: self.gain * value for key, value in xt.items()}


class _TinyEma:
    def __init__(self) -> None:
        self.updates = 0

    def update(self) -> None:
        self.updates += 1


class _TinyWeaver(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wm = _TinyFlow()
        self.rm = torch.nn.Linear(1, 1)
        self.critic = torch.nn.Linear(1, 1)
        self.task_encoder = torch.nn.Linear(1, 1)
        self.task_encoder.feature_dim = 768
        self.full_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.ema = _TinyEma()
        self.full_forward_calls = 0
        self._n_history = 1
        self._use_memory = False
        self._history_noise_std = 0.0
        self._img_keys = ("camera",)
        self._state_loss_scale = 0.1
        self._use_temporal_loss = False
        self._temporal_loss_coeff = 0.1
        self._train_decoder = False

    def encode_obs(self, obs):
        return {"camera": obs["camera"], "states": obs["states"]}

    def sample_noise(self, x1):
        return {key: torch.zeros_like(value) for key, value in x1.items()}

    def sample_timestep(self, batch_size, sequence_length):
        return torch.ones(batch_size, sequence_length)

    def interpolate(self, x1, x0, timestep):
        del x0, timestep
        return x1

    def compute_flow_loss(self, x1, x0, prediction, timestep):
        del timestep
        return (prediction - (x1 - x0)).square()

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        del device_type
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(parameters, lr=learning_rate, betas=betas, weight_decay=weight_decay)

    def forward(self, *, obs, actions, tasks, gt_rewards, memory=None):
        del actions, tasks, memory
        self.full_forward_calls += 1
        prediction = self.full_scale * obs["states"].squeeze(-1)
        loss = (prediction - gt_rewards).square().mean()
        return loss, {"Total Loss": loss.detach()}


def _batch(*, rewards: bool = False, invalid_future: bool = False, invalid_tail: bool = False):
    transition_valid = torch.tensor([[not invalid_future, True, not invalid_tail]], dtype=torch.bool)
    result = {
        "obs": {
            "camera": torch.ones(1, 3, 1, 1),
            "states": torch.ones(1, 3, 1),
        },
        "actions": torch.zeros(1, 3, 1),
        "transition_valid": transition_valid,
        "task": {"text": ["turn on the radio"]},
    }
    if rewards:
        result["rewards"] = torch.ones(1, 3)
    return result


def _move(value, *, device):
    if isinstance(value, Mapping):
        return {key: _move(item, device=device) for key, item in value.items()}
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _tools(saved):
    def save_checkpoint(path, *, model, optimizer, cfg, step, atomic):
        saved.append(
            {
                "path": path,
                "model": model,
                "optimizer": optimizer,
                "cfg": cfg,
                "step": step,
                "atomic": atomic,
            }
        )

    return training_runtime.UpstreamTools(
        move_tensors_to_device=_move,
        save_checkpoint=save_checkpoint,
        get_lr=lambda step, warmup, maximum, max_lr, min_lr: max_lr,
    )


def test_full_mode_calls_upstream_forward_and_requires_real_supervision():
    model = _TinyWeaver()

    loss, _ = training_runtime.compute_loss(model, _batch(rewards=True), mode="full")

    assert loss.requires_grad
    assert model.full_forward_calls == 1
    with pytest.raises(ValueError, match="real rewards"):
        training_runtime.compute_loss(model, _batch(rewards=False), mode="full")


@pytest.mark.parametrize("invalid_field", ["invalid_future", "invalid_tail"])
def test_full_mode_rejects_any_transition_that_frozen_heads_cannot_mask(invalid_field):
    with pytest.raises(ValueError, match="no reward/critic transition mask"):
        training_runtime.compute_loss(
            _TinyWeaver(),
            _batch(rewards=True, **{invalid_field: True}),
            mode="full",
        )


def test_dynamics_optimizer_and_forward_exclude_reward_critic_and_task_heads():
    model = _TinyWeaver()
    optimizer = training_runtime.build_optimizer(
        model,
        mode="dynamics",
        weight_decay=0.01,
        learning_rate=0.1,
        betas=(0.9, 0.99),
        device="cpu",
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    selected = dict(model.named_parameters())

    assert id(selected["wm.gain"]) in optimizer_ids
    assert id(selected["full_scale"]) in optimizer_ids
    assert all(id(parameter) not in optimizer_ids for parameter in model.rm.parameters())
    assert all(id(parameter) not in optimizer_ids for parameter in model.critic.parameters())
    assert all(id(parameter) not in optimizer_ids for parameter in model.task_encoder.parameters())

    loss, log = training_runtime.compute_loss(model, _batch(), mode="dynamics")
    loss.backward()

    assert model.wm.gain.grad is not None
    assert all(parameter.grad is None for parameter in model.rm.parameters())
    assert all(parameter.grad is None for parameter in model.critic.parameters())
    assert all(parameter.grad is None for parameter in model.task_encoder.parameters())
    assert log["valid_future_targets"].item() == 2


def test_train_runs_backward_ema_tiny_overfit_and_atomic_checkpoint(tmp_path):
    model = _TinyWeaver()
    optimizer = training_runtime.build_optimizer(
        model,
        mode="dynamics",
        weight_decay=0.0,
        learning_rate=0.1,
        betas=(0.9, 0.99),
        device="cpu",
    )
    saved = []
    rm_before = model.rm.weight.detach().clone()
    result = training_runtime.train(
        model,
        [_batch()],
        optimizer,
        loop=training_runtime.LoopConfig(
            mode="dynamics",
            device="cpu",
            max_steps=3,
            gradient_accumulation_steps=1,
            max_lr=0.1,
            min_lr=0.1,
            warmup_steps=0,
            weight_decay=0.0,
            betas=(0.9, 0.99),
            checkpoint_every=2,
            tiny_overfit_batches=1,
        ),
        upstream=_tools(saved),
        checkpoint_dir=tmp_path,
        saved_config={"mode": "test"},
    )

    assert result.completed_step == 3
    assert len(result.reports) == 3
    assert result.telemetry["device"] == "cpu"
    assert model.wm.gain.item() != 0.0
    torch.testing.assert_close(model.rm.weight, rm_before)
    assert model.ema.updates == 3
    assert [item["step"] for item in saved] == [2, 3]
    assert all(item["atomic"] is True for item in saved)


def test_resume_restores_optimizer_and_completed_step():
    model = _TinyWeaver()
    optimizer = model.configure_optimizers(0.0, 0.1, (0.9, 0.99), "cpu")
    checkpoint = {"optimizer": optimizer.state_dict(), "step": 7}

    assert training_runtime.restore_optimizer_for_resume(optimizer, checkpoint) == 7
    assert checkpoint == {}

    with pytest.raises(ValueError, match="optimizer and step"):
        training_runtime.restore_optimizer_for_resume(optimizer, {})


def test_loop_config_rejects_invalid_tiny_overfit_values():
    kwargs = {
        "mode": "dynamics",
        "device": "cpu",
        "max_steps": 3,
        "gradient_accumulation_steps": 1,
        "max_lr": 0.1,
        "min_lr": 0.0,
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "betas": (0.9, 0.99),
    }
    with pytest.raises(ValueError, match="tiny_overfit_steps"):
        training_runtime.LoopConfig(**kwargs, tiny_overfit_steps=0)


def test_full_task_may_use_precomputed_clip_features():
    batch = _batch(rewards=True)
    batch["task"] = {"features": torch.zeros(1, 768)}
    model = _TinyWeaver()

    training_runtime.compute_loss(model, batch, mode="full")

    assert model.full_forward_calls == 1
