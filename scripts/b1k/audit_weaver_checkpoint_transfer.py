"""Audit released WEAVER checkpoint identity and B1K 8D-to-23D transfer.

This program is intentionally read-only with respect to training state:
it constructs models, loads the released checkpoint, reports structural
compatibility, and exits. It creates no optimizer and performs no training.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from openpi.weaver import upstream_config
from openpi.weaver import upstream_runtime

SCHEMA_VERSION = "weaver_checkpoint_transfer_audit_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit released WEAVER 8D identity and 8D-to-23D transfer")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to released WEAVER config.yaml.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing released checkpoint.pt.",
    )
    parser.add_argument(
        "--task-encoder-model",
        required=True,
        help="Pinned local CLIP snapshot directory.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write the audit report.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Construction device, normally cuda.",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(value).__name__}")

    return value


def _make_23d_config(source: dict[str, Any]) -> dict[str, Any]:
    """Copy released config and change only state/action dimensionality."""
    released, _ = upstream_config.released_checkpoint_config(source)
    target = copy.deepcopy(released)
    dataset = target.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError("Released config must contain a dataset mapping")

    dataset["n_states"] = 23
    dataset["n_actions"] = 23

    return target


def _parameter_inheritance(
    model: torch.nn.Module,
    *,
    loaded_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Measure inherited parameter mass, not merely key count."""

    parameters = dict(model.named_parameters())
    loaded = set(loaded_keys)

    total_numel = sum(parameter.numel() for parameter in parameters.values())

    inherited_numel = sum(parameters[name].numel() for name in loaded if name in parameters)

    total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters.values())

    inherited_bytes = sum(
        parameters[name].numel() * parameters[name].element_size() for name in loaded if name in parameters
    )

    inherited_parameter_names = tuple(name for name in parameters if name in loaded)

    fresh_parameter_names = tuple(name for name in parameters if name not in loaded)

    return {
        "total_parameter_count": total_numel,
        "inherited_parameter_count": inherited_numel,
        "inherited_parameter_fraction": (inherited_numel / total_numel if total_numel else 0.0),
        "total_parameter_bytes": total_bytes,
        "inherited_parameter_bytes": inherited_bytes,
        "inherited_byte_fraction": (inherited_bytes / total_bytes if total_bytes else 0.0),
        "inherited_parameter_names": list(inherited_parameter_names),
        "fresh_parameter_names": list(fresh_parameter_names),
    }


def _print_gate(
    title: str,
    report: upstream_runtime.CheckpointLoadReport,
) -> None:
    print()
    print(f"===== {title} =====")
    print(f"mode:              {report.mode}")
    print(f"loaded keys:       {len(report.model.loaded_keys)}")
    print(f"shape mismatches:  {len(report.model.shape_mismatches)}")
    print(f"missing source:    {len(report.model.missing_source_keys)}")
    print(f"unexpected source: {len(report.model.unexpected_source_keys)}")
    print(f"EMA strategy:      {report.ema.strategy}")

    for mismatch in report.model.shape_mismatches:
        print(
            "  mismatch: "
            f"{mismatch['key']} "
            f"{tuple(mismatch['source_shape'])} -> "
            f"{tuple(mismatch['target_shape'])}"
        )


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    task_encoder_path = Path(args.task_encoder_model).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not task_encoder_path.is_dir():
        raise FileNotFoundError(task_encoder_path)

    released_config = _read_yaml(config_path)

    released_concrete, inferred_source_fields = upstream_config.released_checkpoint_config(released_config)
    released_contract = upstream_config.architecture_contract(released_concrete)

    if (released_contract.n_states, released_contract.n_actions) != (8, 8):
        raise RuntimeError(
            "Identity gate requires released 8D/8D config, got "
            f"{released_contract.n_states}/{released_contract.n_actions}"
        )

    print("===== released artifacts =====")
    print(f"config:       {config_path}")
    print(f"checkpoint:   {checkpoint_path}")
    print(f"task encoder: {task_encoder_path}")
    print("released dims: " f"{released_contract.n_states} state / " f"{released_contract.n_actions} action")

    # released architecture must load the released checkpoint, including the complete EMA schema.
    print()
    print("===== constructing released 8D identity model =====")

    model_8d = upstream_runtime.build_model(
        released_concrete,
        device=args.device,
        task_encoder_model=task_encoder_path,
    )

    identity_result = upstream_runtime.load_checkpoint_into_model(
        model_8d,
        checkpoint_dir,
        mode=upstream_runtime.IDENTITY_MODE,
        device=args.device,
    )

    identity_report = identity_result.report
    _print_gate("8D IDENTITY GATE", identity_report)

    if identity_report.model.shape_mismatches:
        raise RuntimeError("Identity gate unexpectedly reported shape mismatches")
    if identity_report.model.missing_source_keys:
        raise RuntimeError("Identity gate unexpectedly reported missing source keys")
    if identity_report.model.unexpected_source_keys:
        raise RuntimeError("Identity gate unexpectedly reported unexpected source keys")
    if identity_report.ema.strategy != "loaded_full":
        raise RuntimeError(f"Identity gate used unexpected EMA strategy: {identity_report.ema.strategy}")

    identity_dict = identity_report.to_dict()

    del model_8d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # change only n_states/n_actions from 8 to 23.
    target_config = _make_23d_config(released_config)

    target_concrete = upstream_config.checkpoint_defaults(target_config)
    target_contract = upstream_config.architecture_contract(target_concrete)

    if (target_contract.n_states, target_contract.n_actions) != (23, 23):
        raise RuntimeError(
            "Transfer gate requires target 23D/23D config, got "
            f"{target_contract.n_states}/{target_contract.n_actions}"
        )

    print()
    print("===== constructing B1K 23D transfer model =====")

    model_23d = upstream_runtime.build_model(
        target_config,
        device=args.device,
        task_encoder_model=task_encoder_path,
    )

    transfer_result = upstream_runtime.load_checkpoint_into_model(
        model_23d,
        checkpoint_dir,
        mode=upstream_runtime.B1K_23D_FRESH_MODE,
        device=args.device,
    )

    transfer_report = transfer_result.report
    _print_gate("23D TRANSFER GATE", transfer_report)

    inheritance = _parameter_inheritance(
        model_23d,
        loaded_keys=transfer_report.model.loaded_keys,
    )

    if transfer_report.ema.strategy != "reset_from_target_model":
        raise RuntimeError(f"23D gate used unexpected EMA strategy: {transfer_report.ema.strategy}")

    expected_mismatches = set(upstream_runtime.EXPECTED_8D_TO_23D_MISMATCHES)
    actual_mismatches = {item["key"] for item in transfer_report.model.shape_mismatches}

    if actual_mismatches != expected_mismatches:
        raise RuntimeError(
            "23D mismatch set differs from audited contract: "
            f"expected={sorted(expected_mismatches)}, "
            f"actual={sorted(actual_mismatches)}"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "task_encoder_model": str(task_encoder_path),
            "n_states": released_contract.n_states,
            "n_actions": released_contract.n_actions,
            "inferred_config_fields": list(inferred_source_fields),
        },
        "identity_gate": {
            "passed": True,
            **identity_dict,
        },
        "target": {
            "n_states": target_contract.n_states,
            "n_actions": target_contract.n_actions,
            "config_changes": {
                "dataset.n_states": [8, 23],
                "dataset.n_actions": [8, 23],
            },
        },
        "transfer_gate": {
            "passed": True,
            **transfer_report.to_dict(),
        },
        "inheritance": inheritance,
        "ema": {
            "identity_strategy": identity_report.ema.strategy,
            "transfer_strategy": transfer_report.ema.strategy,
            "transfer_shadow_parameter_count": (transfer_report.ema.shadow_parameter_count),
            "transfer_reset_parameter_count": len(transfer_report.ema.reset_parameter_names),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print()
    print("===== inheritance =====")
    print(
        "parameters: "
        f"{inheritance['inherited_parameter_count']:,} / "
        f"{inheritance['total_parameter_count']:,} "
        f"({inheritance['inherited_parameter_fraction']:.8%})"
    )
    print(
        "bytes:      "
        f"{inheritance['inherited_parameter_bytes']:,} / "
        f"{inheritance['total_parameter_bytes']:,} "
        f"({inheritance['inherited_byte_fraction']:.8%})"
    )

    print()
    print(f"JSON report: {output_path}")
    print("AUDIT PASSED")

    del model_23d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
