from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openpi.weaver import submission_bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(tmp_path: Path, *, profile_name: str = "docs_public_report") -> Path:
    profile = submission_bundle.PROFILES[profile_name]
    root = tmp_path / "standard.public.omega.lab.2026-08-23"
    (root / "json").mkdir(parents=True)
    (root / "videos").mkdir()
    wrapper = root / "wrapper.py"
    wrapper.write_text("class Wrapper:\n    pass\n", encoding="utf-8")
    robot = root / "r1pro.yaml"
    robot.write_text(
        """model: r1pro
name: robot_r1
eval:
  camera_sensor_names:
    head: robot_r1:zed_link:Camera:0
    left_wrist: robot_r1:left_realsense_link:Camera:0
    right_wrist: robot_r1:right_realsense_link:Camera:0
obs_modalities: [proprio, rgb]
proprio_obs: [base_qvel, arm_left_qpos, arm_right_qpos]
sensor_config: {VisionSensor: {}}
controller_config: {base: {name: HolonomicBaseJointController}}
""",
        encoding="utf-8",
    )
    instance = profile.instance_ids[0]
    stem = f"task_alpha_{instance}_0"
    metric = root / "json" / f"{stem}.json"
    metric.write_text(
        json.dumps(
            {
                "task": "task_alpha",
                "instance_id": instance,
                "rollout_id": 0,
                "steps": 5,
                "success": False,
                "q_score": {"final": 0.25},
                "time": {"simulator_steps": 5, "simulator_time": 1.0, "normalized_time": 0.5},
                "agent_distance": {"base": 1.0, "left": 2.0, "right": 3.0},
                "normalized_agent_distance": {"base": 0.1, "left": 0.2, "right": 0.3},
            }
        ),
        encoding="utf-8",
    )
    video = root / "videos" / f"{stem}.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32)
    readme = root / "README.md"
    indices = " ".join(str(value) for value in profile.cli_indices or ())
    readme.write_text(
        "\n".join(
            (
                "python -m omnigibson.eval.eval --task-name TASK",
                f"--instance-indices {indices} --num-rollouts 1",
                "--env-wrapper wrapper.py --robot-config r1pro.yaml",
                "--output-dir outputs --write-video",
                "Docker image: example.invalid/omega:sha256",
            )
        ),
        encoding="utf-8",
    )
    inventory = {
        path.relative_to(root).as_posix(): {"sha256": _sha(path), "size": path.stat().st_size}
        for path in (metric, video)
    }
    manifest = {
        "schema_version": "behavior_2026_compliance_v1",
        "track": "behavior_2026_single_onboard",
        "submission_track": "standard",
        "submission_profile": profile_name,
        "behavior_tag": "v3.9.1",
        "behavior_git_sha": submission_bundle.BEHAVIOR_V391_GIT_SHA,
        "omega_git_sha": "a" * 40,
        "weaver_git_sha": "b" * 40,
        "wrapper": {"path": "wrapper.py", "sha256": _sha(wrapper)},
        "robot_config": {"path": "r1pro.yaml", "sha256": _sha(robot), "action_dim": 23},
        "allowed_observation_keys": ["robot_r1::proprio", "robot_r1::camera::rgb"],
        "task_ids": ["task_alpha"],
        "action_dim": 23,
        "eval_mode": "public_test",
        "instance_indices": list(profile.cli_indices or ()),
        "resolved_instance_ids": list(profile.instance_ids),
        "num_rollouts": 1,
        "selector": {"oracle_used": False, "online_inputs_privileged": False},
        "outputs_unmodified": True,
        "artifact_inventory": inventory,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.mark.parametrize("profile", tuple(submission_bundle.PROFILES))
def test_partial_bundle_passes_each_explicit_public_profile(tmp_path: Path, profile: str) -> None:
    root = _write_bundle(tmp_path, profile_name=profile)
    assert submission_bundle.validate_submission_bundle(root, profile=profile) == ()


def test_full_docs_profile_requires_100_tasks_and_complete_301_to_310_coverage(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    violations = submission_bundle.validate_submission_bundle(root, full_submission=True)
    assert {item.code for item in violations} >= {"task_count", "coverage"}


def test_inventory_detects_post_recording_output_mutation(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    metric = next((root / "json").glob("*.json"))
    metric.write_text(metric.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    violations = submission_bundle.validate_submission_bundle(root)
    assert {item.code for item in violations} >= {"inventory_hash", "inventory_size"}


def test_rejects_score_utils_instance_under_docs_profile(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path, profile_name="score_utils_public")
    violations = submission_bundle.validate_submission_bundle(root, profile="docs_public_report")
    assert {item.code for item in violations} >= {"submission_profile", "instances", "resolved_instances"}


def test_rejects_privileged_selector_and_escaping_wrapper(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selector"]["online_inputs_privileged"] = True
    manifest["wrapper"]["path"] = "../wrapper.py"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    violations = submission_bundle.validate_submission_bundle(root)
    assert {item.code for item in violations} >= {"privileged_online", "wrapper_path"}


def test_rejects_invalid_metric_identity_and_non_mp4_video(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    metric = next((root / "json").glob("*.json"))
    value = json.loads(metric.read_text(encoding="utf-8"))
    value["instance_id"] = 999
    metric.write_text(json.dumps(value), encoding="utf-8")
    video = next((root / "videos").glob("*.mp4"))
    video.write_bytes(b"not an mp4")
    violations = submission_bundle.validate_submission_bundle(root)
    assert {item.code for item in violations} >= {"metric_identity", "video_container"}
