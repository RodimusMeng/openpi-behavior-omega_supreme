"""Fail-fast check for the joint OpenPI + upstream WEAVER serving runtime.

The online selector constructs the JAX policy and the PyTorch WEAVER model in
one Python process.  Importing the top-level packages is not sufficient: this
check imports the concrete encoder and integration modules used by the server
and validates the selected OpenPI action contract without constructing either
GPU model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any


def _version(module: Any) -> str | None:
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-config", default="pi05_b1k_full_abs")
    parser.add_argument("--expected-action-horizon", type=int, default=32)
    parser.add_argument("--expected-model-action-dim", type=int, default=32)
    args = parser.parse_args()

    modules = {
        name: importlib.import_module(name)
        for name in ("diffusers", "jax", "torch", "transformers")
    }

    # Import the concrete symbols exercised by serve_weaver_b1k.py.  Upstream
    # does not export a class named WorldModel from weaver.wm.model; model
    # construction deliberately remains behind openpi.weaver.upstream_runtime.
    from diffusers import AutoencoderKL  # noqa: F401, PLC0415
    from weaver.wm.encoders import ClipEncoder  # noqa: F401, PLC0415
    from weaver.wm.encoders import SD3Encoder  # noqa: F401, PLC0415
    import weaver.wm.model as weaver_model  # noqa: PLC0415

    from openpi.policies import policy_config  # noqa: F401, PLC0415
    from openpi.training import config  # noqa: PLC0415
    from openpi.weaver import upstream_runtime  # noqa: PLC0415

    policy_cfg = config.get_config(args.policy_config)
    actual_horizon = int(policy_cfg.model.action_horizon)
    actual_dim = int(policy_cfg.model.action_dim)
    extra_delta_transform = getattr(policy_cfg.data, "extra_delta_transform", None)
    if actual_horizon != args.expected_action_horizon:
        raise ValueError(
            f"Policy action horizon mismatch: expected {args.expected_action_horizon}, got {actual_horizon}"
        )
    if actual_dim != args.expected_model_action_dim:
        raise ValueError(
            f"Policy model action dimension mismatch: expected {args.expected_model_action_dim}, got {actual_dim}"
        )
    if extra_delta_transform is not False:
        raise ValueError(
            "Borrowed broad B1K checkpoint requires an absolute-action policy config "
            f"with extra_delta_transform=False, got {extra_delta_transform!r}"
        )

    report = {
        "schema_version": "openpi_weaver_combined_runtime_v1",
        "python": sys.executable,
        "versions": {name: _version(module) for name, module in modules.items()},
        "modules": {
            "upstream_runtime": upstream_runtime.__file__,
            "weaver_model": weaver_model.__file__,
        },
        "policy": {
            "config": policy_cfg.name,
            "action_horizon": actual_horizon,
            "model_action_dim": actual_dim,
            "post_transform_b1k_action_dim": 23,
            "extra_delta_transform": extra_delta_transform,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("COMBINED RUNTIME IMPORT GATE: PASSED")


if __name__ == "__main__":
    main()
