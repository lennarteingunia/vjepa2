from __future__ import annotations
import datetime
import logging
import os
import time
from typing import Any, Optional
import warnings

import torch
import tensorboardX
import torch.multiprocessing.spawn
import torch.utils.data.distributed
import tqdm
from models.attentive_pooler import AttentiveClassifier
from src.datasets.full_video_dataset import CSVFullVideoClassifcationDataset
from torch.nn.parallel import DistributedDataParallel as DDP

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
import torch.multiprocessing as mp

_RUN_ID = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"


class TransformDataset(CSVFullVideoClassifcationDataset):

    def __init__(
        self: TransformDataset,
        transforms: video_transforms.Compose,
        *args,
        **kwargs,
    ) -> None:
        super(TransformDataset, self).__init__(*args, **kwargs)

        self._clip_transforms = transforms

    def __getitem__(
        self: TransformDataset,
        index: int,
    ) -> dict[str, str | int | torch.Tensor]:
        sample = super().__getitem__(index)
        sample["data"] = self._clip_transforms(sample["data"])
        return sample


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


def setup(
    rank: int,
    world_size: int,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:

    if logger is None:
        logger = _get_logger(rank)

    if "MASTER_ADDR" not in os.environ:
        logger.warning("MASTER_ADDR environment variable is not set. Defaulting to 'localhost'.")
        os.environ["MASTER_ADDR"] = "localhost"

    if "MASTER_PORT" not in os.environ:
        logger.warning("MASTER_PORT environment variable is not set. Defaulting to '12355'.")
        os.environ["MASTER_PORT"] = "12355"

    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )


def cleanup(rank: int, *, logger: Optional[logging.Logger] = None) -> None:

    if logger is None:
        logger = _get_logger(rank)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    else:
        logger.warning(f"Distributed process group is not initialized on rank {rank}. No cleanup needed.")


def train(
    rank: int,
    world_size: int,
    cfg: dict[str, Any],
    log_dir: str,
    tensorboard_log_dir: str,
) -> None:
    _train(
        rank=rank,
        world_size=world_size,
        log_dir=log_dir,
        tensorboard_log_dir=tensorboard_log_dir,
        **cfg
    )


def _default_backbone_config() -> dict[str, Any]:
    return dict(
        model_name="vjepa2_vit_giant_384"  # Default model name, can be overridden by config
    )


def _default_model_config() -> dict[str, Any]:
    return dict(
        backbone=_default_backbone_config()
    )


def _default_train_data_config() -> dict[str, Any]:
    return dict(
        csv_paths=["/mnt/datasets/wlasl/wlasl/metadata/train100vjepa.csv"],
        frames_per_clip=16,
        frame_step=4,
        stride=8,
        pad_start=False,
        pad_end=True,
    )


def _default_val_data_config() -> dict[str, Any]:
    return dict(
        csv_paths=["/mnt/datasets/wlasl/wlasl/metadata/val100vjepa.csv"],
        frames_per_clip=16,
        frame_step=4,
        stride=8,
        pad_start=False,
        pad_end=True,
    )


def _default_data_config() -> dict[str, Any]:
    return dict(
        csv_sep=" ",
        img_size=224,
        train=_default_train_data_config(),
        val=_default_val_data_config()
    )


def _default_multihead_kwargs() -> list[dict[str, Any]]:
    return [dict(
        final_lr=0.0,
        final_weight_decay=0.01,
        lr=0.005,
        start_lr=0.005,
        warmup=0.0,
        weight_decay=0.01
    )]


def _default_opt_config() -> dict[str, Any]:
    return dict(
        epochs=20,
        batch_size=4,
        multihead_kwargs=_default_multihead_kwargs()
    )


def _default_classifier_config() -> dict[str, Any]:
    return dict(
        num_heads=16,
        depth=4,
    )


def _setup_log_dirs(output_dir: str) -> dict[str, str]:
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."

    log_dir = os.path.join(output_dir, _RUN_ID)
    os.mkdir(log_dir)

    return dict(
        log_dir=log_dir,
        tensorboard_log_dir=log_dir,
    )


def _train(
    rank: int,
    world_size: int,
    log_dir: str,
    tensorboard_log_dir: str,
    model: dict[str, Any] = _default_model_config(),
    data: dict[str, Any] = _default_data_config(),
    optimization: dict[str, Any] = _default_opt_config(),
) -> None:

    assert "NUM_WORKERS" in os.environ, "NUM_WORKERS not set as environment variable. Needed for distributed training."

    # Logging setup

    logger = _get_logger(
        rank,
        log_dir=log_dir
    )
    tensorboard_logger = _get_tensorboard_logger(
        rank=rank,
        tensorboard_log_dir=tensorboard_log_dir
    )

    # DDP Setup

    logger.info(f"Starting training process on rank {rank} with world size {world_size}")
    logger.info(f"Setup distributed environment with rank {rank} and world size {world_size}.")

    setup(rank, world_size, logger=logger)

    # -- MODEL CONFIGURATION --
    backbone_cfg = model.get("backbone", _default_backbone_config())
    model_name = backbone_cfg.get("model_name", "vjepa2_vit_giant_384")

    logger.info(f"Loading model {model_name} from torch hub.")
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", model_name)  # type: ignore
    encoder = encoder.to(rank)
    embed_dim = encoder.embed_dim
    encoder = DDP(encoder, device_ids=[rank], output_device=rank)
    encoder.eval()

    classifier_cfg = model.get("classifier", _default_classifier_config())
    classifiers = [
        DDP(
            AttentiveClassifier(
                embed_dim=embed_dim,
                num_classes=data.get("num_classes", 100),
                use_activation_checkpointing=True,
                **classifier_cfg,
            ).to(rank),
            static_graph=True,
            device_ids=[rank],
            output_device=rank,
        ) for _ in optimization.get("multihead_kwargs", _default_multihead_kwargs())
    ]

    # -- OPTIMIZATION CONFIGURATION --

    # TODO: Make weight decay work.

    optimizers = [
        torch.optim.AdamW(
            params=classifier.parameters(),
            lr=optim_params["lr"],
            weight_decay=optim_params["weight_decay"]
        ) for classifier, optim_params in zip(classifiers, optimization.get("multihead_kwargs", _default_multihead_kwargs()))
    ]

    # -- DATA CONFIGURATION --
    train_config = data.get("train", _default_train_data_config())

    logger.info(f"Indexing and preparing training dataset on rank {rank}")
    train_dataset = TransformDataset(
        sep=data["csv_sep"],
        # TODO: Actually no, I need better transforms here.
        transforms=get_evaluation_transforms(img_size=data["img_size"]),
        **train_config
    )

    logger.info(f"Creating distributed sampler for training on rank {rank}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    logger.info(f"Creating training dataloader in rank {rank}.")
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        num_workers=int(os.environ["NUM_WORKERS"]),
        batch_size=optimization["batch_size"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_config = data.get("val", _default_val_data_config())

    logger.info(f"Indexing and preparing validation dataset on rank {rank}")
    val_dataset = TransformDataset(
        sep=data["csv_sep"],
        transforms=get_evaluation_transforms(img_size=data["img_size"]),
        **val_config
    )

    logger.info(f"Creating distributed sampler for validation on rank {rank}")
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    logger.info(f"Creating validation dataloader on rank {rank}.")
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=optimization["batch_size"],
        num_workers=int(os.environ["NUM_WORKERS"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    for epoch in range(optimization.get("epochs", 20)):
        logger.info(f"Running training epoch {epoch} on rank {rank}.")
        _train_epoch(
            rank=rank,
            world_size=world_size,
            dataloader=train_dataloader,
            encoder=encoder,
            classifiers=classifiers,  # type: ignore
            optimizers=optimizers,
            tensorboard_logger=tensorboard_logger,
            logger=logger,
        )

    logger.info(f"Cleaning up distributed environment for rank {rank}.")
    cleanup(rank, logger=logger)


def _train_epoch(
    rank: int,
    world_size: int,
    dataloader: torch.utils.data.DataLoader,
    encoder: torch.nn.Module,
    classifiers: list[torch.nn.Module],
    optimizers: list,
    *,
    tensorboard_logger: Optional[tensorboardX.SummaryWriter],
    logger: Optional[logging.Logger] = None,
) -> None:

    if logger is None:
        logger = _get_logger(rank=rank)

    logger.info(f"Making new training iterable on rank {rank}.")
    data_iter = iter(dataloader)
    pbar = tqdm.tqdm(enumerate(data_iter), total=len(dataloader))

    loss_fn = torch.nn.CrossEntropyLoss()

    logger.info(f"Starting new training loop on rank {rank}.")
    for idx, batch in pbar:

        data = batch["data"].to(rank)
        label = batch["label"].to(rank)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features = encoder(data)
            outputs = [classifier(features) for classifier in classifiers]
            losses = [loss_fn(output, label) for output in outputs]

        for loss, optimizer in zip(losses, optimizers):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if not tensorboard_logger is None:

            for loss in losses:
                tensorboard_logger.add_scalar("train/loss", scalar_value=loss.item(), global_step=idx)


def main(
    world_size: int = 2,
    fname: str = "configs/wlasl100.yaml",
    output_dir: str = ".logs",
) -> None:
    cfg = _read_cfg(fname)
    log_dirs = _setup_log_dirs(output_dir=output_dir)
    mp.spawn(  # type: ignore
        fn=train,
        args=(world_size, cfg, log_dirs["log_dir"], log_dirs["tensorboard_log_dir"]),
        nprocs=world_size,
        join=True,
    )


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Video Classification Training Script")
    parser.add_argument("--world-size", type=int, default=2, help="Number of processes to spawn")
    parser.add_argument("--output-dir", type=str, default=".logs", help="Root log output dir.")
    parser.add_argument("--fname", type=str, default="configs/wlasl100.yaml", help="Path to the configuration file")
    return parser.parse_args()


def _get_logger(
    rank: int,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__ + f".{rank}")


def _get_tensorboard_logger(
    rank: int,
    tensorboard_log_dir: str,
) -> tensorboardX.SummaryWriter:

    class _EmptyLogger(tensorboardX.SummaryWriter):

        def __getattribute__(self, name: str) -> Any:

            def _pass(*args, **kwargs):
                pass

            return _pass

    if rank == 0:
        return tensorboardX.SummaryWriter(
            log_dir=tensorboard_log_dir,
        )

    return _EmptyLogger(
        log_dir=tensorboard_log_dir
    )


def _read_cfg(fname: str) -> dict:
    import yaml
    with open(fname, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


if __name__ == "__main__":
    args = _parse_args()
    main(
        world_size=args.world_size,
        fname=args.fname,
        output_dir=args.output_dir,
    )
