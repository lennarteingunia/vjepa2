from __future__ import annotations

import argparse
import glob
import logging
import os
import pathlib
import shutil
import warnings

import torch
import yaml

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms

from datasets.full_video_dataset import FullVideoClassifcationDataset


class InferenceDataset(FullVideoClassifcationDataset):

    def __init__(
        self: InferenceDataset,
        root_dir: str,
        *,
        file_endings: list[str] | str = "mp4",
        video_transforms: video_transforms.Compose = video_transforms.Compose([]),
        **kwargs,
    ) -> None:

        if isinstance(file_endings, str):
            file_endings = [file_endings]

        paths = []
        for file_ending in file_endings:
            glob_pattern = os.path.join(root_dir, "**", f"*.{file_ending}")
            paths.extend(glob.glob(glob_pattern))

        empty_labels = [-1] * len(paths)

        super(InferenceDataset, self).__init__(
            paths=paths,
            labels=empty_labels,
            **kwargs
        )

        self._video_transforms = video_transforms

    def __getitem__(self: InferenceDataset, index: int) -> dict[str, str | int | torch.Tensor]:
        sample = super().__getitem__(index)
        sample["data"] = self._video_transforms(sample["data"])
        return sample


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Extract features from a directory of videos into a given output location.")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=64)
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--pad-video-start", action=argparse.BooleanOptionalAction)
    parser.add_argument("--pad-video-end", action=argparse.BooleanOptionalAction)
    parser.add_argument("--file-endings", nargs="+", default=["mp4"], required=False)
    parser.add_argument("--torch-hub-model-name", type=str, default="vjepa2_vit_giant_384")
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--device", type=torch.device,
                        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    return args


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_evaluation_transforms(img_size: int):
    short_side_size = int(256. / 224 * img_size)
    return video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )
    ])


def main(
    model_name: str,
    root_dir: str,
    output_dir: str,
    frames_per_clip: int,
    frame_step: int,
    stride: int,
    pad_start: bool,
    pad_end: bool,
    file_endings: list[str],
    crop_size: int,
    device: torch.device,
) -> None:

    if not os.path.exists(output_dir):
        warnings.warn(f"Output directory {output_dir} does not exist. Creating it.")
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving parameters to file.")
    params_path = os.path.join(output_dir, "params.yaml")
    params = dict(
        model_name=model_name,
        root_dir=root_dir,
        output_dir=output_dir,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        stride=stride,
        pad_video_start=pad_start,
        pad_video_end=pad_end,
        file_endings=file_endings,
        crop_size=crop_size,
        device=device.type
    )
    with open(params_path, "w") as f:
        yaml.dump(params, f, default_flow_style=False)

    logging.info(f"Saving script to output directory.")
    shutil.copyfile(__file__, os.path.join(output_dir, "extract_video_features.py"))

    model, _ = torch.hub.load("facebookresearch/vjepa2", model_name)  # type: ignore
    model = model.to(device)
    model.eval()

    video_transforms = get_evaluation_transforms(img_size=crop_size)

    inference_ds = InferenceDataset(
        root_dir=root_dir,
        file_endings=file_endings,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        stride=stride,
        pad_start=pad_start,
        pad_end=pad_end,
        video_transforms=video_transforms
    )

    inference_dataloader = torch.utils.data.DataLoader(
        dataset=inference_ds,
        batch_size=2,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        drop_last=False
    )

    inference_iter = iter(inference_dataloader)

    # Remember the current highest clip ID per video
    existing_ids = {}

    for batch in inference_iter:

        clip_batch = batch["data"].to(device)

        with torch.inference_mode():
            output_features = model(clip_batch)

        paths = batch["path"]

        for output_feature, path, clip_indices, duration_seconds, pts_seconds in zip(output_features, paths, batch["indices"], batch["duration_seconds"], batch["pts_seconds"]):

            # Create output directory per video if it does not exist
            video_name = os.path.splitext(os.path.basename(path))[0]
            video_output_dir = os.path.join(output_dir, video_name)
            if not os.path.exists(video_output_dir):
                pathlib.Path(video_output_dir).mkdir(parents=True, exist_ok=True)
                existing_ids[video_name] = 0

            # Clip ID are with 12 leading zeros
            clip_id = f"{existing_ids[video_name]:012d}"
            existing_ids[video_name] += 1

            output_path = os.path.join(video_output_dir, f"{clip_id}.pt")
            logging.info(f"Saving output feature to {output_path}")
            torch.save(output_feature, output_path)

            # We are also saving all relevant metadata, i.e. clip indices
            metadata = {
                "clip_indices": clip_indices,
                "duration_seconds": duration_seconds,
                "pts_seconds": pts_seconds
            }
            metadata_path = os.path.join(video_output_dir, f"{clip_id}.yaml")
            logging.info(f"Saving metadata to {metadata_path}")
            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_arguments()
    main(
        model_name=args.torch_hub_model_name,
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        stride=args.stride,
        pad_start=args.pad_video_start,
        pad_end=args.pad_video_end,
        file_endings=args.file_endings,
        crop_size=args.crop_size,
        device=args.device,
    )
