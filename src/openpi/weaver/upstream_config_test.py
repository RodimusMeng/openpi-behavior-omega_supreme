import copy

import pytest

from openpi.weaver import upstream_config


def _released_config() -> dict:
    return {
        "defaults": {
            "horizon": 8,
            "n_history": 2,
            "n_memory_frames": 6,
            "t_memory": 5,
            "use_precomputed_features": True,
            "dataset": {
                "name": "DROID",
                "path": "/droid",
                "img_keys": ["wrist_left", "exterior_1_left"],
                "image_size": [192, 320],
                "n_states": 8,
                "n_actions": 8,
                "normalize": True,
            },
            "im_encoder": {
                "name": "stabilityai/stable-diffusion-3-medium-diffusers",
                "spatial_size": 2,
            },
            "model": {"n_embed": 1536},
            "training": {"max_steps": 1_000_000},
            "inference": {"pyramid_stagger_width": 1},
        },
        "debug": {"training": {"max_steps": 2}},
    }


def test_behavior_config_changes_only_explicit_embodiment_and_data_fields() -> None:
    source = _released_config()
    original = copy.deepcopy(source)

    target, report = upstream_config.build_behavior_config(
        source,
        dataset_root="/b1k/numeric",
        norm_stats_path="/b1k/norm_stats.json",
    )

    assert source == original
    assert report.source.n_states == report.source.n_actions == 8
    assert report.target.n_states == report.target.n_actions == 23
    assert report.changed_fields == ("n_states", "n_actions")
    assert target["dataset"]["name"] == "BEHAVIOR"
    assert target["dataset"]["img_keys"] == ["wrist_left", "exterior_1_left"]
    assert target["dataset"]["reward_mode"] == "disabled"
    assert target["dataset"]["state_representation"] == "r1pro_compact_state_v1"
    assert target["dataset"]["action_representation"] == "r1pro_compact_state_delta_v1"
    assert target["training"]["dynamics_only"] is True
    assert target["model"] == source["defaults"]["model"]
    assert target["im_encoder"] == source["defaults"]["im_encoder"]
    assert target["inference"] == source["defaults"]["inference"]


def test_saved_concrete_checkpoint_config_is_supported() -> None:
    source = _released_config()["defaults"]
    contract = upstream_config.architecture_contract(source)
    assert contract.image_keys == ("wrist_left", "exterior_1_left")
    assert contract.image_size == (192, 320)
    assert contract.n_states == 8
    assert contract.spatial_size == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"image_keys": ("exterior_1_left", "wrist_left")}, "two-view order"),
        ({"image_keys": ("wrist_left", "exterior_1_left", "wrist_right")}, "two-view order"),
        ({"n_states": 8}, "R1Pro transfer requires"),
        ({"n_actions": 8}, "R1Pro transfer requires"),
    ],
)
def test_first_transfer_rejects_unreviewed_architecture_variants(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        upstream_config.build_behavior_config(
            _released_config(),
            dataset_root="/b1k",
            norm_stats_path="/b1k/norm.json",
            **kwargs,
        )


def test_contract_rejects_missing_or_invalid_shape_fields() -> None:
    source = _released_config()
    del source["defaults"]["dataset"]["n_actions"]
    with pytest.raises(ValueError, match="missing"):
        upstream_config.architecture_contract(source)

    source = _released_config()
    source["defaults"]["n_history"] = True
    with pytest.raises(ValueError, match="n_history"):
        upstream_config.architecture_contract(source)


def test_behavior_transfer_accepts_legacy_release_with_both_action_fields_omitted() -> None:
    source = _released_config()
    del source["defaults"]["dataset"]["n_states"]
    del source["defaults"]["dataset"]["n_actions"]

    target, report = upstream_config.build_behavior_config(
        source,
        dataset_root="/b1k",
        norm_stats_path="/b1k/norm.json",
    )

    assert report.source.n_states == report.source.n_actions == 8
    assert report.target.n_states == report.target.n_actions == 23
    assert report.changed_fields == ("n_states", "n_actions")
    assert report.inferred_source_fields == ("n_states", "n_actions")
    assert target["dataset"]["n_states"] == 23
    assert target["dataset"]["n_actions"] == 23


def test_released_checkpoint_config_materializes_only_valid_legacy_release_metadata() -> None:
    source = _released_config()
    del source["defaults"]["dataset"]["n_states"]
    del source["defaults"]["dataset"]["n_actions"]
    del source["defaults"]["use_precomputed_features"]

    concrete, inferred = upstream_config.released_checkpoint_config(source)

    assert concrete["dataset"]["n_states"] == 8
    assert concrete["dataset"]["n_actions"] == 8
    assert concrete["use_precomputed_features"] is True
    assert inferred == ("n_states", "n_actions", "use_precomputed_features")

    source["defaults"]["horizon"] = 12
    with pytest.raises(ValueError, match="frozen DROID release architecture"):
        upstream_config.released_checkpoint_config(source)


def test_released_checkpoint_config_rejects_explicit_non_release_precomputed_mode() -> None:
    source = _released_config()
    source["defaults"]["use_precomputed_features"] = False

    with pytest.raises(ValueError, match="use_precomputed_features=True"):
        upstream_config.released_checkpoint_config(source)


def test_behavior_transfer_infers_one_missing_legacy_action_metadata_field() -> None:
    source = _released_config()
    del source["defaults"]["dataset"]["n_states"]

    _, report = upstream_config.build_behavior_config(
        source,
        dataset_root="/b1k",
        norm_stats_path="/b1k/norm.json",
    )
    assert report.inferred_source_fields == ("n_states",)

    source = _released_config()
    del source["defaults"]["dataset"]["n_actions"]
    _, report = upstream_config.build_behavior_config(
        source,
        dataset_root="/b1k",
        norm_stats_path="/b1k/norm.json",
    )
    assert report.inferred_source_fields == ("n_actions",)


def test_behavior_transfer_rejects_partial_metadata_with_non_release_dimension() -> None:
    source = _released_config()
    del source["defaults"]["dataset"]["n_states"]
    source["defaults"]["dataset"]["n_actions"] = 23

    with pytest.raises(ValueError, match="frozen 8D release value"):
        upstream_config.build_behavior_config(
            source,
            dataset_root="/b1k",
            norm_stats_path="/b1k/norm.json",
        )


def test_behavior_transfer_rejects_missing_metadata_for_non_release_architecture() -> None:
    source = _released_config()
    del source["defaults"]["dataset"]["n_states"]
    del source["defaults"]["dataset"]["n_actions"]
    source["defaults"]["horizon"] = 12

    with pytest.raises(ValueError, match="frozen DROID release architecture"):
        upstream_config.build_behavior_config(
            source,
            dataset_root="/b1k",
            norm_stats_path="/b1k/norm.json",
        )
