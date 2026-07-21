"""Integration for the legacy YOLOv8 API used by the original project.

The imports match codebases that expose ``ultralytics.yolo.*``. This file is
kept separate so the core modules can be tested without installing a specific
Ultralytics fork.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from copy import copy
from typing import Any, Optional

import numpy as np
import torch

from daq_yolo.losses.aqfl import AQFocalLossWithLogits, aligned_box_iou
from daq_yolo.modules.d_ema import D_EMA
from daq_yolo.nms.qha_nms import QHANMSConfig, qha_non_max_suppression

try:
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.yolo import v8
    from ultralytics.yolo.data import build_dataloader
    from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
    from ultralytics.yolo.engine.trainer import BaseTrainer
    from ultralytics.yolo.utils import RANK, colorstr
    from ultralytics.yolo.utils.loss import BboxLoss
    from ultralytics.yolo.utils.ops import xywh2xyxy
    from ultralytics.yolo.utils.plotting import plot_images, plot_labels, plot_results
    from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
    from ultralytics.yolo.utils.torch_utils import de_parallel
except ImportError as exc:  # pragma: no cover - depends on user's fork
    raise ImportError(
        "legacy_yolov8.py targets the older Ultralytics layout used by the "
        "uploaded train.py (ultralytics.yolo.*). Install/use that fork, or "
        "port the small Loss and NMS hooks to your installed version."
    ) from exc


def register_custom_modules() -> None:
    """Expose ``D_EMA`` to YAML evaluation in ``ultralytics.nn.tasks``.

    The parse_model channel-injection branch described in
    ``docs/INTEGRATION_GUIDE.md`` is still required.
    """
    import ultralytics.nn.tasks as tasks

    tasks.D_EMA = D_EMA


class QHANMSPatch(AbstractContextManager["QHANMSPatch"]):
    """Temporarily replace legacy Ultralytics NMS with QHA-NMS."""

    def __init__(self, config: Optional[QHANMSConfig] = None) -> None:
        self.config = config or QHANMSConfig()
        self._original = None

    def __enter__(self) -> "QHANMSPatch":
        from ultralytics.yolo.utils import ops

        self._original = ops.non_max_suppression

        def patched(*args: Any, **kwargs: Any):
            kwargs["qha_config"] = self.config
            return qha_non_max_suppression(*args, **kwargs)

        ops.non_max_suppression = patched
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        from ultralytics.yolo.utils import ops

        if self._original is not None:
            ops.non_max_suppression = self._original
        self._original = None


class DAQDetectionLoss:
    """YOLOv8 detection loss with AQFL replacing classification BCE."""

    def __init__(self, model) -> None:
        device = next(model.parameters()).device
        hyp = model.args
        detect_head = model.model[-1]

        self.hyp = hyp
        self.stride = detect_head.stride
        self.nc = detect_head.nc
        self.no = detect_head.no
        self.reg_max = detect_head.reg_max
        self.device = device
        self.use_dfl = detect_head.reg_max > 1

        self.aqfl = AQFocalLossWithLogits(
            alpha=float(getattr(hyp, "aqfl_alpha", 0.25)),
            gamma0=float(getattr(hyp, "aqfl_gamma0", 2.0)),
            lambda_=float(getattr(hyp, "aqfl_lambda", 1.0)),
            beta=float(getattr(hyp, "aqfl_beta", 1.0)),
            reduction="none",
            formulation=str(getattr(hyp, "aqfl_formulation", "paper")),
            detach_iou=bool(getattr(hyp, "aqfl_detach_iou", False)),
        ).to(device)

        min_memory = getattr(hyp, "min_memory", 0)
        roll_out_thr = min_memory if min_memory > 1 else 64 if min_memory else 0
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            roll_out_thr=roll_out_thr,
        )
        self.bbox_loss = BboxLoss(detect_head.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(detect_head.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        image_indices = targets[:, 0]
        _, counts = image_indices.unique(return_counts=True)
        output = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        for image_index in range(batch_size):
            matches = image_indices == image_index
            count = matches.sum()
            if count:
                output[image_index, :count] = targets[matches, 1:]
        output[..., 1:5] = xywh2xyxy(output[..., 1:5].mul_(scale_tensor))
        return output

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch, anchors, channels = pred_dist.shape
            pred_dist = (
                pred_dist.view(batch, anchors, 4, channels // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, predictions, batch):
        loss = torch.zeros(3, device=self.device)
        features = predictions[1] if isinstance(predictions, tuple) else predictions
        concatenated = torch.cat(
            [feature.view(features[0].shape[0], self.no, -1) for feature in features], dim=2
        )
        pred_distribution, pred_scores = concatenated.split((self.reg_max * 4, self.nc), dim=1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distribution = pred_distribution.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        image_size = torch.tensor(
            features[0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(features, self.stride, 0.5)

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            dim=1,
        )
        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=image_size[[1, 0, 1, 0]],
        )
        gt_labels, gt_boxes = targets.split((1, 4), dim=2)
        gt_mask = gt_boxes.sum(2, keepdim=True).gt_(0)
        pred_boxes = self.bbox_decode(anchor_points, pred_distribution)

        _, target_boxes, target_scores, foreground_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_boxes.detach() * stride_tensor).type(gt_boxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_boxes,
            gt_mask,
        )
        target_boxes /= stride_tensor
        target_scores_sum = target_scores.sum().clamp_min(1.0)

        aligned_iou = aligned_box_iou(pred_boxes, target_boxes)
        aligned_iou = aligned_iou * foreground_mask.to(aligned_iou.dtype)
        classification = self.aqfl(
            pred_scores,
            target_scores.to(dtype),
            iou_scores=aligned_iou,
            positive_mask=foreground_mask,
        )
        loss[1] = classification.sum() / target_scores_sum

        if foreground_mask.any():
            loss[0], loss[2] = self.bbox_loss(
                pred_distribution,
                pred_boxes,
                anchor_points,
                target_boxes,
                target_scores,
                target_scores_sum,
                foreground_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


class DAQDetectionTrainer(BaseTrainer):
    """Minimal legacy trainer using :class:`DAQDetectionLoss`."""

    def get_dataloader(self, dataset_path, batch_size, mode="train", rank=0):
        grid_size = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if self.args.v5loader:
            return create_dataloader(
                path=dataset_path,
                imgsz=self.args.imgsz,
                batch_size=batch_size,
                stride=grid_size,
                hyp=vars(self.args),
                augment=mode == "train",
                cache=self.args.cache,
                pad=0 if mode == "train" else 0.5,
                rect=self.args.rect or mode == "val",
                rank=rank,
                workers=self.args.workers,
                close_mosaic=self.args.close_mosaic != 0,
                prefix=colorstr(f"{mode}: "),
                shuffle=mode == "train",
                seed=self.args.seed,
            )[0]
        return build_dataloader(
            self.args,
            batch_size,
            img_path=dataset_path,
            stride=grid_size,
            rank=rank,
            mode=mode,
            rect=mode == "val",
            names=self.data["names"],
        )[0]

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_model(self, cfg=None, weights=None, verbose=True):
        register_custom_modules()
        model = DetectionModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return v8.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args)
        )

    def criterion(self, predictions, batch):
        if not hasattr(self, "compute_loss"):
            self.compute_loss = DAQDetectionLoss(de_parallel(self.model))
        return self.compute_loss(predictions, batch)

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{name}" for name in self.loss_names]
        if loss_items is None:
            return keys
        return dict(zip(keys, [round(float(value), 5) for value in loss_items]))

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch", "GPU_mem", *self.loss_names, "Instances", "Size"
        )

    def plot_training_samples(self, batch, iteration):
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{iteration}.jpg",
        )

    def plot_metrics(self):
        plot_results(file=self.csv)

    def plot_training_labels(self):
        boxes = np.concatenate([label["bboxes"] for label in self.train_loader.dataset.labels], 0)
        classes = np.concatenate([label["cls"] for label in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, classes.squeeze(), names=self.data["names"], save_dir=self.save_dir)
