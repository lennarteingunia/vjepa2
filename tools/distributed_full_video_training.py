from __future__ import annotations
from abc import ABC, abstractmethod
import datetime
import logging
import math
import os
import time
from typing import Any, Literal, Optional
import warnings

from overrides import override
import torch
import tensorboardX
import torch.multiprocessing.spawn
import torch.utils.data.distributed
import torchmetrics
import torchmetrics.classification
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


class _Scheduler(ABC):

    def __init__(
        self: _Scheduler,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.optimizer = optimizer
        self._reset()

    @property
    def global_step(self: _Scheduler) -> int:
        return self._step

    @property
    def optimizer(self: _Scheduler) -> torch.optim.Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self: _Scheduler, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer

    def _reset(self: _Scheduler) -> None:
        self._step = 0

    @abstractmethod
    def step(self: _Scheduler) -> None:
        raise NotImplementedError(f"Method {_Scheduler.step.__name__} has not been implemented.")


class WarmUpCosineSchedule(_Scheduler):

    def __init__(
        self: WarmUpCosineSchedule,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        *,
        mode: Literal["lr", "weight_decay"] = "lr",
    ) -> None:
        super(WarmUpCosineSchedule, self).__init__(optimizer=optimizer)

        self.T_max = T_max
        self.mode = mode

    @override
    def step(self: WarmUpCosineSchedule) -> None:
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_value = group[f"mc_ref_{self.mode}"]
            final_value = group[f"mc_final_{self.mode}"]
            start_value = group[f"mc_start_{self.mode}"]
            warmup_steps = group[f"mc_warmup_steps_{self.mode}"]
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, T_max))
                new_value = start_value + progress * (ref_value - start_value)
            else:
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_value = max(final_value, final_value + (ref_value - final_value)
                                * 0.5 * (1.0 + math.cos(math.pi * progress)))
            group[self.mode] = new_value


class CosineSchedule(_Scheduler):

    def __init__(
        self: CosineSchedule,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        *,
        mode: Literal["lr", "weight_decay"] = "weight_decay"
    ) -> None:
        super(CosineSchedule, self).__init__(optimizer=optimizer)
        self.T_max = T_max
        self.mode = mode

    def step(self: CosineSchedule) -> None:

        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:

            ref_value = group[f"mc_ref_{self.mode}"]
            final_value = group[f"mc_final_{self.mode}"]

            new_value = final_value + (ref_value - final_value) * 0.5 * (1.0 + math.cos(math.pi * progress))

            if final_value <= ref_value:
                new_value = max(final_value, new_value)

            else:
                new_value = min(final_value, new_value)

            group[self.mode] = new_value


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

    # -- OPTIMIZATION CONFIGURATION --

    optimizers, lr_schedulers, wd_schedulers, grad_scalers = [], [], [], []
    for classifier, optim_params in zip(classifiers, optimization.get("multihead_kwargs", _default_multihead_kwargs())):
        param_groups = [
            {
                "params": (p for _, p in classifier.named_parameters()),
                # TODO: Iterations per epoch?
                "mc_warmup_steps_lr": int(optim_params.get("warmup") * len(train_dataloader)),
                "mc_start_lr": optim_params.get("start_lr"),
                "mc_ref_lr": optim_params.get("lr"),
                "mc_final_lr": optim_params.get("final_lr"),
                "mc_ref_weight_decay": optim_params.get("weight_decay"),
                "mc_final_weight_decay": optim_params.get("final_weight_decay")
            }
        ]

        optimizer = torch.optim.AdamW(params=param_groups)
        lr_scheduler = WarmUpCosineSchedule(
            optimizer=optimizer,
            T_max=int(len(train_dataloader) * optimization.get("epochs", 20)),
            mode="lr",
        )
        wd_scheduler = CosineSchedule(
            optimizer=optimizer,
            T_max=int(len(train_dataloader) * optimization.get("epochs", 20)),
            mode="weight_decay"
        )
        grad_scaler = torch.amp.grad_scaler.GradScaler()
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)
        wd_schedulers.append(wd_scheduler)
        grad_scalers.append(grad_scaler)

    validation_metrics = dict(
        accuracy=torchmetrics.classification.MulticlassAccuracy(
            num_classes=data.get("num_classes", 100)
        ),
        auroc=torchmetrics.classification.MulticlassAUROC(
            num_classes=data.get("num_classes", 100),
            thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        ),
        cm=torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=data.get("num_classes", 100),
        )
    )

    validation_metrics = {name: metric.to(rank) for name, metric in validation_metrics.items()}

    def _val_epoch_fn(epoch: int): return _val_epoch(
        rank=rank,
        world_size=world_size,
        data_loader=val_dataloader,
        encoder=encoder,
        classifiers=classifiers,  # type: ignore
        metrics=validation_metrics,
        tensorboard_logger=tensorboard_logger,
        logger=logger,
        epoch=epoch
    )

    for epoch in range(optimization.get("epochs", 20)):
        _val_epoch_fn(epoch=epoch)
        logger.info(f"Running training epoch {epoch} on rank {rank}.")
        _train_epoch(
            rank=rank,
            world_size=world_size,
            dataloader=train_dataloader,
            encoder=encoder,
            classifiers=classifiers,  # type: ignore
            optimizers=optimizers,
            schedulers=lr_schedulers + wd_schedulers,
            grad_scalers=grad_scalers,
            tensorboard_logger=tensorboard_logger,
            logger=logger,
            epoch=epoch,
        )
    _val_epoch_fn(epoch=optimization.get("epochs", 20) - 1)

    logger.info(f"Cleaning up distributed environment for rank {rank}.")
    cleanup(rank, logger=logger)


@torch.no_grad()
def _val_epoch(
    rank: int,
    world_size: int,
    data_loader: torch.utils.data.DataLoader,
    encoder: torch.nn.Module,
    classifiers: list[torch.nn.Module],
    metrics: dict[str, torchmetrics.Metric] = dict(),
    *,
    tensorboard_logger: Optional[tensorboardX.SummaryWriter] = None,
    logger: Optional[logging.Logger] = None,
    epoch: int = 0,
) -> None:

    if logger is None:
        logger = _get_logger(rank=rank)

    logger.info(f"Making new validation iterable on rank {rank}.")
    data_iter = iter(data_loader)
    pbar = tqdm.tqdm(enumerate(data_iter), total=len(data_loader))

    logger.info(f"Resetting validation metrics on epoch {epoch}.")
    for metric in metrics.values():
        metric.reset()

    per_classifier_metrics = [{name: metric.clone() for name, metric in metrics.items()} for _ in classifiers]

    for idx, batch in pbar:

        data = batch["data"].to(rank)
        label = batch["label"].to(rank)

        features = encoder(data)
        outputs = [classifier(features) for classifier in classifiers]

        for metrics, output in zip(per_classifier_metrics, outputs):
            for metric in metrics.values():
                metric.update(output, label)

    if not tensorboard_logger is None:

        for idx, metrics in enumerate(per_classifier_metrics):
            for metric_name, metric in metrics.items():
                tensorboard_logger.add_scalar(f"val/{idx}/{metric_name}", metric.compute().item(), global_step=epoch)


def _train_epoch(
    rank: int,
    world_size: int,
    dataloader: torch.utils.data.DataLoader,
    encoder: torch.nn.Module,
    classifiers: list[torch.nn.Module],
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[_Scheduler],
    grad_scalers: list[torch.amp.grad_scaler.GradScaler],
    *,
    tensorboard_logger: Optional[tensorboardX.SummaryWriter] = None,
    logger: Optional[logging.Logger] = None,
    epoch: int = 0,
) -> None:

    if logger is None:
        logger = _get_logger(rank=rank)

    logger.info(f"Making new training iterable on rank {rank}.")
    data_iter = iter(dataloader)
    pbar = tqdm.tqdm(enumerate(data_iter), total=len(dataloader))

    loss_fn = torch.nn.CrossEntropyLoss()

    logger.info(f"Starting new training loop on rank {rank}.")
    for idx, batch in pbar:

        for optimizer in optimizers:
            optimizer.zero_grad()

        for scheduler in schedulers:
            scheduler.step()

        data = batch["data"].to(rank)
        label = batch["label"].to(rank)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features = encoder(data)
            outputs = [classifier(features) for classifier in classifiers]
            losses = [loss_fn(output, label) for output in outputs]

        for loss, optimizer, scaler in zip(losses, optimizers, grad_scalers):
            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer)
            scaler.update()

        if not tensorboard_logger is None:

            for loss in losses:
                tensorboard_logger.add_scalar("train/loss", scalar_value=loss.item(),
                                              global_step=idx + len(dataloader) * epoch)


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
