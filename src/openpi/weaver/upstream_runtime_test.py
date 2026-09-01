import json
from types import SimpleNamespace

import pytest

from openpi.weaver import upstream_runtime


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _config(dim: int) -> dict:
    return {
        "defaults": {
            "horizon": 8,
            "n_history": 2,
            "n_memory_frames": 6,
            "t_memory": 5,
            "use_precomputed_features": True,
            "use_compile": True,
            "use_activation_checkpointing": True,
            "dataset": {
                "img_keys": ["wrist_left", "exterior_1_left"],
                "image_size": [192, 320],
                "n_states": dim,
                "n_actions": dim,
                "sample_aux_from_left_right": False,
            },
            "im_encoder": {
                "name": "stabilityai/stable-diffusion-3-medium-diffusers",
                "spatial_size": 2,
                "future_encoder_knob": "preserved",
            },
            "model": {
                "n_embed": 1536,
                "use_sprint": True,
                "sprint_drop_ratio": 0.5,
                "future_model_knob": {"kept": True},
            },
            "inference": {
                "pyramid_stagger_width": 1,
                "future_inference_knob": "kept",
            },
        }
    }


class _FakeEncoder:
    spatial_size = 2
    feature_dim = 64


class _FakeEmaInner:
    def __init__(self):
        self.shadow_params = []
        self.optimization_step = 17


class _FakeEma:
    def __init__(self, state=None):
        self.ema = _FakeEmaInner()
        self._state = state or {"decay": 0.9, "shadow_params": [_FakeTensor((2, 2))]}
        self.loaded = False
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state
        self.loaded = True


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class _FakeParameter(_FakeTensor):
    requires_grad = True


class _FakeModel:
    def __init__(self, *, state=None, ema_state=None, **kwargs):
        self.kwargs = kwargs
        self.ema = _FakeEma(ema_state)
        self._state = state or {"shared.weight": _FakeTensor((2, 2))}
        self.loaded_state = None
        self.strict = None
        self.device = None
        self._parameters = (
            ("shared.weight", _FakeParameter((2, 2))),
            ("head.weight", _FakeParameter((1, 2))),
        )

    def to(self, device):
        self.device = device
        return self

    def state_dict(self):
        return self._state

    def load_state_dict(self, state, *, strict):
        self.loaded_state = state
        self.strict = strict
        return SimpleNamespace(missing_keys=(), unexpected_keys=())

    def named_parameters(self):
        return iter(self._parameters)


def _build_api(calls):
    def get_encoder(*, config, image_size, device):
        calls["encoder"] = {"config": config, "image_size": image_size, "device": device}
        return _FakeEncoder(), False

    def get_task_encoder(*, config, device):
        calls["task_encoder"] = {"config": config, "device": device}
        return SimpleNamespace(feature_dim=768)

    class ModelType(_FakeModel):
        def __init__(self, **kwargs):
            calls["model_kwargs"] = kwargs
            super().__init__(**kwargs)

    return upstream_runtime._UpstreamApi(  # noqa: SLF001
        get_encoder=get_encoder,
        get_task_encoder=get_task_encoder,
        model_type=ModelType,
        dict_to_namespace=_namespace,
        load_checkpoint=None,
    )


@pytest.mark.parametrize("dim", [8, 23])
def test_build_model_reuses_complete_upstream_construction_for_identity_and_b1k(monkeypatch, dim):
    calls = {}
    monkeypatch.setattr(upstream_runtime, "_load_upstream_api", lambda: _build_api(calls))

    model = upstream_runtime.build_model(_config(dim), device="cpu")

    assert model.device == "cpu"
    assert model.ema.device == "cpu"
    kwargs = calls["model_kwargs"]
    assert kwargs["img_keys"] == ["wrist_left", "exterior_1_left"]
    assert kwargs["n_states"] == kwargs["n_actions"] == dim
    assert kwargs["n_history"] == 2
    assert kwargs["n_horizon"] == 8
    assert kwargs["n_memory_frames"] == 6
    assert kwargs["t_memory"] == 5
    assert kwargs["use_precomputed_features"] is True
    assert kwargs["image_size"] == (192, 320)
    assert kwargs["config"].use_sprint is True
    assert kwargs["config"].future_model_knob.kept is True
    assert kwargs["inference_config"].future_inference_knob == "kept"
    assert calls["encoder"]["config"].future_encoder_knob == "preserved"
    assert calls["task_encoder"] == {"config": None, "device": "cpu"}


def test_build_model_preserves_configured_view_count_knob(monkeypatch):
    calls = {}
    config = _config(23)
    config["defaults"]["dataset"]["img_keys"].append("wrist_right")
    monkeypatch.setattr(upstream_runtime, "_load_upstream_api", lambda: _build_api(calls))

    upstream_runtime.build_model(config, device="cpu")

    assert calls["model_kwargs"]["img_keys"] == ["wrist_left", "exterior_1_left", "wrist_right"]


def test_build_model_passes_pinned_snapshot_to_upstream_clip_encoder(monkeypatch, tmp_path):
    calls = {}
    snapshot = tmp_path / "clip"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    class FakeClipEncoder:
        feature_dim = 768

        def __init__(self, *, model_name, device):
            calls["clip_encoder"] = {"model_name": model_name, "device": device}

    monkeypatch.setattr(upstream_runtime, "_load_upstream_api", lambda: _build_api(calls))
    monkeypatch.setattr(upstream_runtime, "_load_clip_encoder_type", lambda: FakeClipEncoder)

    upstream_runtime.build_model(_config(8), device="cpu", task_encoder_model=snapshot)

    assert calls["clip_encoder"] == {"model_name": str(snapshot.resolve()), "device": "cpu"}
    assert "task_encoder" not in calls


@pytest.mark.parametrize("value", ["", "   "])
def test_build_model_rejects_empty_pinned_snapshot_before_image_encoder(monkeypatch, value):
    calls = {}
    monkeypatch.setattr(upstream_runtime, "_load_upstream_api", lambda: _build_api(calls))

    with pytest.raises(ValueError, match="must be non-empty"):
        upstream_runtime.build_model(_config(8), device="cpu", task_encoder_model=value)

    assert "encoder" not in calls


def test_build_model_rejects_incomplete_pinned_snapshot_before_image_encoder(monkeypatch, tmp_path):
    calls = {}
    snapshot = tmp_path / "clip"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(upstream_runtime, "_load_upstream_api", lambda: _build_api(calls))

    with pytest.raises(FileNotFoundError, match="neither model"):
        upstream_runtime.build_model(_config(8), device="cpu", task_encoder_model=snapshot)

    assert "encoder" not in calls


class _FakeTransferReport:
    def __init__(
        self,
        *,
        shape_mismatches=(),
        missing_source_keys=(),
        unexpected_source_keys=(),
    ):
        self.loaded_keys = ("shared.weight",)
        self.shape_mismatches = tuple(shape_mismatches)
        self.missing_source_keys = tuple(missing_source_keys)
        self.unexpected_source_keys = tuple(unexpected_source_keys)
        self.source_key_rewrites = ()
        self.target_key_rewrites = ()
        self.is_identity_compatible = not (
            self.shape_mismatches or self.missing_source_keys or self.unexpected_source_keys
        )


class _FakeTransferApi:
    def __init__(self, report):
        self.report = report
        self.require_identity = None
        self.reset_called = False

    def adapt_state_dict(self, source, target, *, require_identity):
        self.require_identity = require_identity
        return target, self.report

    def reset_ema_from_model(self, ema_inner, model):
        self.reset_called = True
        named = tuple(model.named_parameters())
        ema_inner.shadow_params = [_FakeTensor(parameter.shape) for _, parameter in named]
        ema_inner.optimization_step = 0
        return SimpleNamespace(
            parameter_names=tuple(name for name, _ in named),
            shadow_shapes=tuple(parameter.shape for _, parameter in named),
        )


def _loader_api(checkpoint):
    def load_checkpoint(path, *, device, weights_only):
        assert path.endswith("checkpoint-dir")
        assert device == "cpu"
        assert weights_only is True
        return checkpoint

    return upstream_runtime._UpstreamApi(None, None, None, None, load_checkpoint)  # noqa: SLF001


def _checkpoint(*, ema_shape=(2, 2), include_resume=True):
    value = {
        "model": {"shared.weight": _FakeTensor((2, 2))},
        "ema": {"decay": 0.9, "shadow_params": [_FakeTensor(ema_shape)]},
    }
    if include_resume:
        value.update({"optimizer": {"state": "optimizer"}, "step": 123})
    return value


@pytest.mark.parametrize(
    ("mode", "resumed"),
    [
        (upstream_runtime.IDENTITY_MODE, False),
        (upstream_runtime.RESUME_MODE, True),
    ],
)
def test_strict_identity_and_resume_load_complete_model_and_ema(monkeypatch, mode, resumed):
    checkpoint = _checkpoint()
    transfer = _FakeTransferApi(_FakeTransferReport())
    monkeypatch.setattr(upstream_runtime, "_load_upstream_api", lambda: _loader_api(checkpoint))
    monkeypatch.setattr(upstream_runtime, "_load_transfer_api", lambda: transfer)
    model = _FakeModel(ema_state={"decay": 0.9, "shadow_params": [_FakeTensor((2, 2))]})

    result = upstream_runtime.load_checkpoint_into_model(
        model,
        "checkpoint-dir",
        mode=mode,
        device="cpu",
    )

    assert model.strict is True
    assert model.ema.loaded
    assert transfer.require_identity is True
    assert result.resumed is resumed
    assert result.report.ema.strategy == "loaded_full"
    assert result.report.model.shape_mismatches == ()
    if resumed:
        assert result.checkpoint["optimizer"] == {"state": "optimizer"}
        assert result.checkpoint["step"] == 123
    else:
        assert "optimizer" not in result.checkpoint
        assert "step" not in result.checkpoint
    json.dumps(result.report.to_dict())


def _cross_mismatches():
    values = []
    for key in upstream_runtime.EXPECTED_8D_TO_23D_MISMATCHES:
        source_shape, target_shape = ((8, 4), (23, 4)) if key == "wm.out_prj.states.weight" else ((4, 8), (4, 23))
        values.append(SimpleNamespace(key=key, source_shape=source_shape, target_shape=target_shape))
    return tuple(values)


def test_cross_dimension_transfer_requires_nine_shapes_and_resets_all_ema(monkeypatch):
    checkpoint = _checkpoint()
    transfer = _FakeTransferApi(_FakeTransferReport(shape_mismatches=_cross_mismatches()))
    monkeypatch.setattr(upstream_runtime, "_load_upstream_api", lambda: _loader_api(checkpoint))
    monkeypatch.setattr(upstream_runtime, "_load_transfer_api", lambda: transfer)
    model = _FakeModel()

    result = upstream_runtime.load_checkpoint_into_model(
        model,
        "checkpoint-dir",
        mode=upstream_runtime.B1K_23D_FRESH_MODE,
        device="cpu",
    )

    assert model.strict is True
    assert not model.ema.loaded
    assert transfer.require_identity is False
    assert transfer.reset_called
    assert result.resumed is False
    assert "optimizer" not in result.checkpoint
    assert "step" not in result.checkpoint
    assert result.report.ema.strategy == "reset_from_target_model"
    assert result.report.ema.reset_parameter_names == ("shared.weight", "head.weight")
    assert len(result.report.model.shape_mismatches) == 9
    assert model.ema.ema.optimization_step == 0
    json.dumps(result.report.to_dict())


def test_cross_dimension_rejects_any_tenth_or_missing_name(monkeypatch):
    checkpoint = _checkpoint()
    mismatches = (*_cross_mismatches(), SimpleNamespace(key="extra.weight", source_shape=(4, 8), target_shape=(4, 23)))
    transfer = _FakeTransferApi(_FakeTransferReport(shape_mismatches=mismatches))
    monkeypatch.setattr(upstream_runtime, "_load_upstream_api", lambda: _loader_api(checkpoint))
    monkeypatch.setattr(upstream_runtime, "_load_transfer_api", lambda: transfer)

    with pytest.raises(upstream_runtime.CheckpointContractError, match="exactly the nine"):
        upstream_runtime.load_checkpoint_into_model(
            _FakeModel(),
            "checkpoint-dir",
            mode=upstream_runtime.B1K_23D_FRESH_MODE,
            device="cpu",
        )


def test_identity_rejects_incomplete_ema_and_resume_requires_optimizer(monkeypatch):
    transfer = _FakeTransferApi(_FakeTransferReport())
    monkeypatch.setattr(upstream_runtime, "_load_transfer_api", lambda: transfer)
    wrong_ema = _checkpoint(ema_shape=(3, 3))
    monkeypatch.setattr(upstream_runtime, "_load_upstream_api", lambda: _loader_api(wrong_ema))
    with pytest.raises(upstream_runtime.CheckpointContractError, match="EMA tensor shape differs"):
        upstream_runtime.load_checkpoint_into_model(
            _FakeModel(),
            "checkpoint-dir",
            mode=upstream_runtime.IDENTITY_MODE,
            device="cpu",
        )

    no_optimizer = _checkpoint(include_resume=False)
    monkeypatch.setattr(upstream_runtime, "_load_upstream_api", lambda: _loader_api(no_optimizer))
    with pytest.raises(upstream_runtime.CheckpointContractError, match="optimizer and step"):
        upstream_runtime.load_checkpoint_into_model(
            _FakeModel(),
            "checkpoint-dir",
            mode=upstream_runtime.RESUME_MODE,
            device="cpu",
        )
