

from __future__ import annotations
import logging
from typing import Any, Optional
import warnings

import numpy as np
import torch
import torchcodec
import tqdm


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
        metadata: Optional[list[dict[str, Any]]] = None,
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

            pbar.set_description(desc=f"{path}")

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

        self._samples = samples
        logging.info(f"Created {FullVideoClassifcationDataset.__name__} with a total of {len(self._samples)} samples.")

    def __len__(self: FullVideoClassifcationDataset) -> int:
        return len(self._samples)

    def __getitem__(self: FullVideoClassifcationDataset, index: int) -> dict[str, str | torch.Tensor | int]:
        sample = self._samples[index]
        video_decoder = torchcodec.decoders.VideoDecoder(sample["path"])
        clip = video_decoder.get_frames_at(sample["indices"])
        return {**sample, "data": clip.data, "duration_seconds": clip.duration_seconds, "pts_seconds": clip.pts_seconds}

    def _convert_labels(self: FullVideoClassifcationDataset, labels: list[str]) -> list[int]:
        label_set = set(labels)
        sorted_labels = sorted(label_set)
        return [sorted_labels.index(label) for label in labels]


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
