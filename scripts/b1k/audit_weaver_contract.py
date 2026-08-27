"""Print the source-derived R1Pro/WEAVER integration contract as JSON."""

import dataclasses
import json

from openpi.weaver import b1k_contract


def contract_payload() -> dict:
    contract = b1k_contract.build_r1pro_contract()
    return {
        "schema_version": 1,
        "omega_base_commit": "06e5675a4625912e3fc25d017cd571002d2123d9",
        "weaver_upstream": {
            "url": "https://github.com/arnavkj1995/WEAVER",
            "commit": "3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e",
            "checkpoint_family": "arnavkj1995/WEAVER/WEAVER",
            "checkpoint_downloaded": False,
        },
        "robot_key": contract.robot_key,
        "state_dim": contract.state_dim,
        "action_dim": contract.action_dim,
        "state_components": [dataclasses.asdict(component) for component in contract.state_components],
        "action_components": [dataclasses.asdict(component) for component in contract.action_components],
        "camera_views": [dataclasses.asdict(view) for view in contract.camera_views],
        "delta_mappings": contract.delta_mappings,
        "timing": {
            "b1k_action_hz": 30,
            "b1k_render_hz": 30,
            "b1k_physics_hz": 120,
            "initial_world_model_hz": 5,
            "sampling_rule": "nearest real timestamp",
        },
        "weaver_boundary": {
            "candidate_shape": "(N,T,23)",
            "action_space": "post-output-transform environment commands (policy-aligned v1)",
            "released_discount_factor": 0.995,
            "future_states": "shape/device/dtype placeholders in upstream inference",
        },
    }


def main() -> None:
    print(json.dumps(contract_payload(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
