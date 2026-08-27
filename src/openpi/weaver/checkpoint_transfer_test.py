import types

import pytest
import torch

from openpi.weaver import checkpoint_transfer

DIMENSION_DEPENDENT_KEYS = (
    "wm.inp_prj.states.c_fc.weight",
    "wm.inp_prj.actions.c_fc.weight",
    "wm.mem_inp_prj.states.c_fc.weight",
    "wm.mem_inp_prj.actions.c_fc.weight",
    "wm.out_prj.states.weight",
    "rm.net.inp_prj.states.c_fc.weight",
    "rm.net.inp_prj.actions.c_fc.weight",
    "critic.net.inp_prj.states.c_fc.weight",
    "critic.net.inp_prj.actions.c_fc.weight",
)

VIEW_KEY_TEMPLATES = (
    "wm.inp_prj.{view}.weight",
    "wm.inp_prj.{view}.bias",
    "wm.mem_inp_prj.{view}.weight",
    "wm.mem_inp_prj.{view}.bias",
    "wm.out_prj.{view}.weight",
    "rm.net.inp_prj.{view}.weight",
    "rm.net.inp_prj.{view}.bias",
    "critic.net.inp_prj.{view}.weight",
    "critic.net.inp_prj.{view}.bias",
)


def test_identity_transfer_normalizes_wrapper_prefixes():
    source_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source_bias = torch.arange(2, dtype=torch.float32)
    source = {
        "module._orig_mod.layer.weight": source_weight,
        "_orig_mod.module.layer.bias": source_bias,
    }
    target = {
        "layer.weight": torch.zeros_like(source_weight),
        "layer.bias": torch.zeros_like(source_bias),
    }

    merged, report = checkpoint_transfer.adapt_state_dict(source, target, require_identity=True)

    assert report.is_identity_compatible
    assert report.loaded_keys == ("layer.weight", "layer.bias")
    assert len(report.source_key_rewrites) == 2
    assert merged["layer.weight"] is source_weight
    assert merged["layer.bias"] is source_bias


def test_cross_dimension_transfer_keeps_exactly_nine_target_initializations():
    source = {"shared.weight": torch.ones(2, 2)}
    target = {"shared.weight": torch.zeros(2, 2)}
    target_initializations = {}

    for key in DIMENSION_DEPENDENT_KEYS:
        if key == "wm.out_prj.states.weight":
            source[key] = torch.ones(8, 4)
            target[key] = torch.full((23, 4), 9.0)
        else:
            source[key] = torch.ones(4, 8)
            target[key] = torch.full((4, 23), 9.0)
        target_initializations[key] = target[key]

    merged, report = checkpoint_transfer.adapt_state_dict(source, target)

    assert merged["shared.weight"] is source["shared.weight"]
    assert tuple(item.key for item in report.shape_mismatches) == DIMENSION_DEPENDENT_KEYS
    assert not report.missing_source_keys
    assert not report.unexpected_source_keys
    for key, initialization in target_initializations.items():
        assert merged[key] is initialization

    with pytest.raises(checkpoint_transfer.IdentityTransferError) as exc_info:
        checkpoint_transfer.adapt_state_dict(source, target, require_identity=True)
    assert exc_info.value.report == report


def test_two_to_three_view_transfer_reports_new_view_and_global_shapes():
    source_views = ("wrist_left", "exterior_1_left")
    target_views = (*source_views, "right_wrist")
    source = {}
    target = {}

    for view in source_views:
        for template in VIEW_KEY_TEMPLATES:
            source[template.format(view=view)] = torch.ones(2, 2)
    for view in target_views:
        for template in VIEW_KEY_TEMPLATES:
            target[template.format(view=view)] = torch.zeros(2, 2)

    source.update(
        {
            "wm._base_rope_ids": torch.arange(480),
            "rm.net.out_prj.c_fc.weight": torch.ones(4, 20),
            "critic.net.out_prj.c_fc.weight": torch.ones(4, 20),
        }
    )
    target.update(
        {
            "wm._base_rope_ids": torch.arange(720),
            "rm.net.out_prj.c_fc.weight": torch.zeros(4, 24),
            "critic.net.out_prj.c_fc.weight": torch.zeros(4, 24),
        }
    )

    _, report = checkpoint_transfer.adapt_state_dict(source, target)

    assert set(report.missing_source_keys) == {template.format(view="right_wrist") for template in VIEW_KEY_TEMPLATES}
    assert tuple(item.key for item in report.shape_mismatches) == (
        "wm._base_rope_ids",
        "rm.net.out_prj.c_fc.weight",
        "critic.net.out_prj.c_fc.weight",
    )
    assert not report.unexpected_source_keys


def test_prefix_normalization_collision_is_rejected():
    source = {
        "module.layer.weight": torch.ones(1),
        "_orig_mod.layer.weight": torch.ones(1),
    }

    with pytest.raises(checkpoint_transfer.KeyNormalizationError, match="both normalize"):
        checkpoint_transfer.adapt_state_dict(source, {"layer.weight": torch.zeros(1)})


def test_reset_ema_from_model_uses_target_trainable_parameter_order_and_clones():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1, bias=False))
    model[0].bias.requires_grad_(requires_grad=False)
    ema = types.SimpleNamespace(shadow_params=[torch.tensor([-1.0])], optimization_step=37)

    report = checkpoint_transfer.reset_ema_from_model(ema, model)

    expected = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert report.parameter_names == tuple(name for name, _ in expected)
    assert report.shadow_shapes == tuple(tuple(parameter.shape) for _, parameter in expected)
    assert ema.optimization_step == 0
    assert len(ema.shadow_params) == len(expected)
    for shadow, (_, parameter) in zip(ema.shadow_params, expected, strict=True):
        torch.testing.assert_close(shadow, parameter)
        assert shadow.data_ptr() != parameter.data_ptr()

    first_shadow = ema.shadow_params[0].clone()
    with torch.no_grad():
        expected[0][1].add_(1)
    torch.testing.assert_close(ema.shadow_params[0], first_shadow)
