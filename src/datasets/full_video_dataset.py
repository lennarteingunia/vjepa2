

from __future__ import annotations
import logging
import os
import pathlib
from typing import Any, Callable, Hashable, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.utils.data.distributed
import torchcodec
import tqdm

from datasets.utils.dataloader import MonitoredDataset
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from src.datasets.utils.dataloader import NondeterministicDataLoader

_GLOBAL_SEED = 0
logger = logging.getLogger()


def make_csv_full_video_classification_dataset(
    data_paths: list[str],
    frames_per_clip: int,
    frame_step: int,
    stride: int,
    batch_size: int,
    drop_last: bool = True,
    pin_memory: bool = True,
    pad_start: bool = False,
    pad_end: bool = True,
    rank: int = 0,
    world_size: int = 1,
    datasets_weights: Optional[list[float]] = None,
    log_dir: Optional[str | pathlib.Path] = None,
    collator: Optional[Callable] = None,
    deterministic: bool = True,
    num_workers: int = 10,
    persistent_workers: bool = True,
) -> tuple[CSVFullVideoClassifcationDataset | MonitoredDataset, torch.utils.data.DataLoader | NondeterministicDataLoader, torch.utils.data.distributed.DistributedSampler | DistributedWeightedSampler]:

    dataset = CSVFullVideoClassifcationDataset(
        csv_paths=data_paths,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        stride=stride,
        pad_start=pad_start,
        pad_end=pad_end,
    )

    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)  # type: ignore
        log_dir.mkdir(parents=True, exist_ok=True)
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0
        )

    logger.info("CSVFullVideoClassifcationDataset created")

    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )

    logger.info("CSVFullVideoClassificationDataset unsupervised dataloader created")
    return dataset, data_loader, dist_sampler


class FullVideoClassifcationDataset(torch.utils.data.Dataset):

    def __init__(
        self: FullVideoClassifcationDataset,
        paths: list[str],
        labels: list[str] | list[int],
        *,
        frames_per_clip: int = 64,
        frame_step: int = 4,
        stride: int = 4,
        pad_start: bool = False,
        pad_end: bool = False,
        metadata: Optional[list[dict[Hashable, Any]]] = None,
    ) -> None:

        assert len(paths) == len(labels), "Not every video seems to have a label or vice-versa."

        if not metadata is None:
            assert len(paths) == len(metadata), "You can either provide a set of meta data per path, or none at all."

        if any(isinstance(label, str) for label in labels):
            labels = self._convert_labels(labels)  # type: ignore

        samples = []

        if metadata is None:
            metadata = [{}] * len(paths)

        pbar = tqdm.tqdm(enumerate(zip(paths, labels, metadata)), total=len(paths))

        for idx, (path, label, mdata_entry) in pbar:

            pbar.set_description(desc=f"Indexing: {path}")

            video_decoder = torchcodec.decoders.VideoDecoder(path)

            if video_decoder.metadata.num_frames is None:
                warnings.warn(
                    f"Video at {path} could not be decoded or does not contain any frames and is being skipped.")
                continue

            clip_indices = _calculate_clip_indices(
                num_frames=video_decoder.metadata.num_frames,
                frames_per_clip=frames_per_clip,
                frame_step=frame_step,
                stride=stride,
                pad_start=pad_start,
                pad_end=pad_end
            )

            samples.extend([
                {
                    "path": path,
                    "label": label,
                    "indices": indices,
                    "metadata": mdata_entry,
                } for indices in clip_indices
            ])

        self.samples = samples
        logging.info(f"Created {FullVideoClassifcationDataset.__name__} with a total of {len(self.samples)} samples.")

    @property
    def samples(self: FullVideoClassifcationDataset) -> list[dict[str, Any]]:
        return self._samples

    @samples.setter
    def samples(self: FullVideoClassifcationDataset, samples: list[dict[str, Any]]) -> None:
        samples = list(sorted(samples, key=lambda sample: (sample["path"], sample["indices"])))
        self._samples = samples

    def __len__(self: FullVideoClassifcationDataset) -> int:
        return len(self.samples)

    def __getitem__(self: FullVideoClassifcationDataset, index: int) -> dict[str, str | torch.Tensor | int]:
        sample = self.samples[index]
        video_decoder = torchcodec.decoders.VideoDecoder(sample["path"])
        clip = video_decoder.get_frames_at(sample["indices"])
        return {**sample, "data": clip.data, "duration_seconds": clip.duration_seconds, "pts_seconds": clip.pts_seconds}

    def _convert_labels(self: FullVideoClassifcationDataset, labels: list[str]) -> list[int]:
        label_set = set(labels)
        sorted_labels = sorted(label_set)
        return [sorted_labels.index(label) for label in labels]


class CSVFullVideoClassifcationDataset(FullVideoClassifcationDataset):

    def __init__(
        self: CSVFullVideoClassifcationDataset,
        csv_paths: list[str],
        *,
        sep: str = " ",
        **kwargs,
    ) -> None:

        df = None

        for csv_path in csv_paths:

            assert csv_path.endswith(".csv"), f"Path {csv_path} does not end with .csv"
            assert os.path.exists(csv_path), f"Path {csv_path} does not exist."

            new_df = pd.read_csv(
                csv_path,
                sep=sep,
                header=None,
            )

            new_df.columns = ["path", "label"] + [f"meta_{i}" for i in range(len(new_df.columns) - 2)]

            if df is None:
                df = new_df
            else:
                df = pd.concat([df, new_df], ignore_index=True)

        assert df is not None, "No data found in the provided CSV files."

        paths = df["path"].tolist()
        labels = df["label"].tolist()
        metadata = df[[col for col in df.columns if col.startswith("meta_")]].to_dict(orient="records")

        if len(metadata) == 0:
            metadata = None

        super(CSVFullVideoClassifcationDataset, self).__init__(
            paths=paths,
            labels=labels,
            metadata=metadata,
            **kwargs,
        )


def _calculate_clip_indices(
    num_frames: int,
    frames_per_clip: int,
    frame_step: int,
    stride: int,
    pad_start: bool = False,
    pad_end: bool = False,
) -> list[list[int]]:

    start_indices = np.arange(0, num_frames, step=stride)

    all_indices = []

    for start in start_indices:

        end = min(int(start + frames_per_clip * frame_step), num_frames)
        clip_indices = np.arange(start, end, step=frame_step)

        if pad_start:

            clip_indices = np.pad(
                clip_indices,
                pad_width=(frames_per_clip - len(clip_indices), 0),
                mode="constant",
                constant_values=clip_indices[0]
            )

        if pad_end:

            clip_indices = np.pad(
                clip_indices,
                pad_width=(0, frames_per_clip - len(clip_indices)),
                mode="constant",
                constant_values=clip_indices[-1]
            )

        if len(clip_indices) < frames_per_clip:
            continue

        clip_indices = clip_indices.tolist()
        all_indices.append(clip_indices)

    return all_indices


if __name__ == "__main__":

    dataset = CSVFullVideoClassifcationDataset(
        csv_paths=[
            "/mnt/datasets/wlasl/wlasl/metadata/train100vjepa.csv"
        ],
    )
