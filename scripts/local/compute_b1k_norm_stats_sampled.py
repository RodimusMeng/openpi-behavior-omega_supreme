from __future__ import annotations

import pathlib
import random
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import tqdm
import tyro

import openpi.transforms as _transforms
import openpi.training.config as _config
from openpi.policies import b1k_policy
from openpi.shared import normalize


def resolve_asset_id(data_config: _config.DataConfig) -> str:
    asset_id = data_config.asset_id or data_config.repo_id

    if asset_id is None:
        raise ValueError("Data config must have asset_id or repo_id")

    if isinstance(asset_id, list):
        if not asset_id:
            raise ValueError("Empty asset_id/repo_id")
        return asset_id[0]

    return asset_id


def scalar_column(table: pa.Table, name: str, dtype):
    arr = table[name].combine_chunks()

    return np.asarray(
        arr.to_numpy(zero_copy_only=False),
        dtype=dtype,
    )


def vector_column(table: pa.Table, name: str, dtype):
    """
    Efficiently convert Arrow list/fixed-size-list column
    to a dense numpy array without creating millions of
    Python lists.
    """
    arr = table[name].combine_chunks()

    if pa.types.is_fixed_size_list(arr.type):
        width = arr.type.list_size

    elif pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        offsets = np.asarray(
            arr.offsets.to_numpy(zero_copy_only=False)
        )

        widths = np.diff(offsets)

        if len(widths) == 0:
            raise ValueError(f"Empty column: {name}")

        if not np.all(widths == widths[0]):
            raise ValueError(
                f"{name} is variable-length; expected fixed vectors"
            )

        width = int(widths[0])

    else:
        raise TypeError(
            f"{name}: expected list-like Arrow type, got {arr.type}"
        )

    values = np.asarray(
        arr.values.to_numpy(zero_copy_only=False),
        dtype=dtype,
    )

    return values.reshape(len(arr), width)


def main(
    config_name: str,
    max_frames: int = 100_000,
    seed: int = 42,
    stats_batch_size: int = 4096,
    output_dir: str | None = None,
):
    t0 = perf_counter()

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = _config.get_config(config_name)

    data_config = config.data.create(
        config.assets_dirs,
        config.model,
    )

    if data_config.dataset_root is None:
        raise ValueError("B1K dataset_root is required")

    horizon = config.model.action_horizon

    # --------------------------------------------------------
    # Reuse EXACT official B1K state/action transforms.
    #
    # data_transforms.inputs:
    #
    #   B1KInputs
    #   MappedDeltaActions
    #
    # We manually do the state part of B1KInputs so images
    # are never touched, then reuse everything after it.
    # --------------------------------------------------------

    input_transforms = data_config.data_transforms.inputs

    if not input_transforms:
        raise ValueError("No B1K input transforms")

    b1k_input = input_transforms[0]

    if not isinstance(b1k_input, b1k_policy.B1KInputs):
        raise TypeError(
            f"Expected B1KInputs, got {type(b1k_input)}"
        )

    robot_config = b1k_input.robot_config

    post_transforms = list(input_transforms[1:])
    post_transform = _transforms.compose(post_transforms)

    print(
        "Post transforms:",
        [type(x).__name__ for x in post_transforms],
    )

    # --------------------------------------------------------
    # Arrow numeric dataset only.
    # --------------------------------------------------------

    data_path = pathlib.Path(data_config.dataset_root) / "data"

    dataset = pads.dataset(
        data_path,
        format="parquet",
    )

    total_rows = dataset.count_rows()

    print(f"Total dataset rows: {total_rows:,}")

    if max_frames > total_rows:
        max_frames = total_rows

    # --------------------------------------------------------
    # Uniform frame sampling WITHOUT replacement.
    #
    # Python random.sample(range(...)) avoids constructing a
    # 210M-element permutation.
    # --------------------------------------------------------

    rng = random.Random(seed)

    sampled_indices = np.asarray(
        rng.sample(
            range(total_rows),
            max_frames,
        ),
        dtype=np.int64,
    )

    sampled_indices.sort()

    print(f"Sampled current frames: {len(sampled_indices):,}")
    print(f"Seed: {seed}")

    # --------------------------------------------------------
    # For every sampled current frame i, we need:
    #
    #   i, i+1, ..., i+31
    #
    # because official B1K training constructs a 32-action
    # future horizon.
    #
    # These can cross an episode boundary. We read them first,
    # then reproduce official terminal-action padding below.
    # --------------------------------------------------------

    offsets = np.arange(
        horizon,
        dtype=np.int64,
    )

    raw_window_indices = (
        sampled_indices[:, None]
        + offsets[None, :]
    )

    in_bounds = raw_window_indices < total_rows

    window_indices = np.minimum(
        raw_window_indices,
        total_rows - 1,
    )

    # Deduplicate because nearby sampled frames can share
    # future-action rows.
    flat_indices = window_indices.reshape(-1)

    unique_indices, inverse = np.unique(
        flat_indices,
        return_inverse=True,
    )

    print(
        f"Unique numeric rows to read: "
        f"{len(unique_indices):,}"
    )

    t_read0 = perf_counter()

    # One numeric Arrow read.
    #
    # No RGB / depth columns.
    table = dataset.take(
        unique_indices,
        columns=[
            "episode_index",
            "frame_index",
            "observation.state",
            "action",
        ],
    )

    t_read1 = perf_counter()

    print(
        f"Arrow numeric read: "
        f"{t_read1 - t_read0:.3f} s"
    )

    # --------------------------------------------------------
    # Convert Arrow -> numpy.
    # --------------------------------------------------------

    episode_u = scalar_column(
        table,
        "episode_index",
        np.int64,
    )

    frame_u = scalar_column(
        table,
        "frame_index",
        np.int64,
    )

    raw_state_u = vector_column(
        table,
        "observation.state",
        np.float32,
    )

    raw_action_u = vector_column(
        table,
        "action",
        np.float32,
    )

    inverse = inverse.reshape(
        max_frames,
        horizon,
    )

    episode_window = episode_u[inverse]
    frame_window = frame_u[inverse]

    # Current state only.
    raw_state = raw_state_u[
        inverse[:, 0]
    ]

    # Full action horizon.
    actions = raw_action_u[
        inverse
    ].copy()

    # --------------------------------------------------------
    # Verify that Arrow global row adjacency really corresponds
    # to temporal adjacency inside each episode.
    #
    # This is important: if dataset fragment ordering did
    # something surprising, abort instead of silently producing
    # wrong norm stats.
    # --------------------------------------------------------

    current_episode = episode_window[:, 0]
    current_frame = frame_window[:, 0]

    same_episode = (
        episode_window
        == current_episode[:, None]
    )

    valid = in_bounds & same_episode

    if not np.all(valid[:, 0]):
        raise RuntimeError(
            "Current sampled row is unexpectedly invalid"
        )

    expected_frame = (
        current_frame[:, None]
        + offsets[None, :]
    )

    temporal_error = (
        valid
        & (frame_window != expected_frame)
    )

    if np.any(temporal_error):
        i, k = np.argwhere(
            temporal_error
        )[0]

        raise RuntimeError(
            "Arrow row order is not temporal-contiguous at "
            f"sample={i}, horizon={k}: "
            f"episode={current_episode[i]}, "
            f"current_frame={current_frame[i]}, "
            f"observed_frame={frame_window[i, k]}, "
            f"expected_frame={expected_frame[i, k]}"
        )

    # Ensure valid positions form one prefix:
    #
    # True True True False False ...
    #
    invalid_seen = np.maximum.accumulate(
        ~valid,
        axis=1,
    )

    if np.any(valid & invalid_seen):
        raise RuntimeError(
            "Episode validity mask is not prefix-contiguous"
        )

    # --------------------------------------------------------
    # Official terminal padding.
    #
    # If future horizon runs past episode end, LeRobot repeats
    # the final action of that episode.
    # --------------------------------------------------------

    valid_count = valid.sum(axis=1)

    last_valid_pos = valid_count - 1

    last_action = actions[
        np.arange(max_frames),
        last_valid_pos,
    ]

    actions = np.where(
        valid[..., None],
        actions,
        last_action[:, None, :],
    )

    # --------------------------------------------------------
    # Official 61D proprio -> robot state extraction.
    #
    # Function is vectorized over leading dimensions.
    # --------------------------------------------------------

    states = b1k_policy.extract_state_from_proprio(
        raw_state,
        robot_config,
    ).astype(
        np.float32,
        copy=False,
    )

    print("state shape:", states.shape)
    print("actions shape:", actions.shape)

    # --------------------------------------------------------
    # RunningStats.
    #
    # Work in chunks so RunningStats never needs a giant
    # temporary flattened action tensor.
    # --------------------------------------------------------

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    t_stats0 = perf_counter()

    for start in tqdm.tqdm(
        range(
            0,
            max_frames,
            stats_batch_size,
        ),
        desc="Computing stats",
    ):
        end = min(
            start + stats_batch_size,
            max_frames,
        )

        batch = {
            "state": states[start:end],
            "actions": actions[start:end].copy(),
        }

        # e.g. MappedDeltaActions
        batch = post_transform(batch)

        stats["state"].update(
            np.asarray(batch["state"])
        )

        stats["actions"].update(
            np.asarray(batch["actions"])
        )

    t_stats1 = perf_counter()

    norm_stats = {
        key: stat.get_statistics()
        for key, stat in stats.items()
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    asset_id = resolve_asset_id(data_config)

    if output_dir is None:
        output_path = (
            config.assets_dirs
            / asset_id
        )
    else:
        output_path = pathlib.Path(
            output_dir
        ).resolve()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Writing stats to: {output_path}")

    normalize.save(
        output_path,
        norm_stats,
    )

    # Keep exact frame sample for reproducibility.
    np.save(
        output_path
        / f"sample_indices_n{max_frames}_seed{seed}.npy",
        sampled_indices,
    )

    t1 = perf_counter()

    print()
    print("========== SUMMARY ==========")
    print(f"config:       {config_name}")
    print(f"dataset rows: {total_rows:,}")
    print(f"sampled:      {max_frames:,}")
    print(f"horizon:      {horizon}")
    print(f"seed:         {seed}")
    print(
        f"numeric read: "
        f"{t_read1 - t_read0:.3f} s"
    )
    print(
        f"stats loop:   "
        f"{t_stats1 - t_stats0:.3f} s"
    )
    print(
        f"TOTAL:        "
        f"{t1 - t0:.3f} s"
    )
    print("=============================")


if __name__ == "__main__":
    tyro.cli(main)