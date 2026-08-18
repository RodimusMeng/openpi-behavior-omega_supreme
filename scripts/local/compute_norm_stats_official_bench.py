"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""
from time import perf_counter
import numpy as np
import tqdm
import tyro
import pathlib
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def resolve_asset_id(data_config: _config.DataConfig) -> str:
    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Data config must have an asset_id or repo_id")
    if isinstance(asset_id, list):
        if not asset_id:
            raise ValueError("Data config asset_id/repo_id list cannot be empty")
        return asset_id[0]
    return asset_id


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_b1k_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_b1k_dataset(data_config=data_config, action_horizon=action_horizon)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches

def main(
    config_name: str,
    max_frames: int | None = None,
    output_dir: str | None = None,
):
    t0 = perf_counter()

    config = _config.get_config(config_name)
    data_config = config.data.create(
        config.assets_dirs,
        config.model,
    )

    t1 = perf_counter()

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            max_frames,
        )
    elif data_config.dataset_root is not None:
        data_loader, num_batches = create_b1k_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.num_workers,
            max_frames,
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )

    t2 = perf_counter()

    keys = ["state", "actions"]
    stats = {
        key: normalize.RunningStats()
        for key in keys
    }

    for batch in tqdm.tqdm(
        data_loader,
        total=num_batches,
        desc="Computing stats",
    ):
        for key in keys:
            stats[key].update(
                np.asarray(batch[key])
            )

    t3 = perf_counter()

    norm_stats = {
        key: stat.get_statistics()
        for key, stat in stats.items()
    }

    asset_id = resolve_asset_id(data_config)

    output_path = (
        pathlib.Path(output_dir)
        if output_dir is not None
        else config.assets_dirs / asset_id
    )

    normalize.save(output_path, norm_stats)

    t4 = perf_counter()

    frames = num_batches * config.batch_size

    print()
    print("========== TIMING ==========")
    print(f"config:       {t1 - t0:.3f} s")
    print(f"loader init:  {t2 - t1:.3f} s")
    print(f"stats loop:   {t3 - t2:.3f} s")
    print(f"save:         {t4 - t3:.3f} s")
    print(f"TOTAL:        {t4 - t0:.3f} s")
    print(f"frames:       {frames}")
    print(f"end2end fps:  {frames / (t4 - t0):.3f}")
    print(f"loop fps:     {frames / (t3 - t2):.3f}")
    print("============================")


if __name__ == "__main__":
    tyro.cli(main)
