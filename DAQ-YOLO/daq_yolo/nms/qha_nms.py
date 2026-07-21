"""Quality-Augmented Heterogeneous Adaptive NMS (QHA-NMS).

The implementation is intentionally pure PyTorch because torchvision NMS only
accepts one scalar IoU threshold, whereas the paper defines one threshold per
candidate box.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor


def box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Pairwise IoU for ``xyxy`` boxes."""
    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[1] != 4 or boxes2.shape[1] != 4:
        raise ValueError("boxes1 and boxes2 must have shape [N, 4] and [M, 4]")
    area1 = ((boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0)).prod(1)
    area2 = ((boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0)).prod(1)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(2)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp_min(eps)


def xywh2xyxy(boxes: Tensor) -> Tensor:
    """Convert center-width-height boxes to corner boxes."""
    output = boxes.clone()
    output[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    output[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    output[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    output[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return output


@dataclass(frozen=True)
class QHANMSConfig:
    """QHA-NMS hyperparameters."""

    base_iou: float = 0.45
    min_iou: float = 0.30
    max_iou: float = 0.70
    density_beta: float = 0.20
    quality_weights: Tuple[float, float, float] = (0.60, 0.30, 0.10)
    ideal_aspect_ratio: float = 1.0
    max_aspect_deviation: float = 5.0
    boundary_lambda: float = 1.0
    threshold_rule: Literal["candidate", "selected", "min", "mean"] = "min"
    localization_quality_mode: Literal["agreement", "confidence"] = "agreement"
    density_chunk_size: int = 512

    def __post_init__(self) -> None:
        if not (0 <= self.min_iou <= self.base_iou <= self.max_iou <= 1):
            raise ValueError("Require 0 <= min_iou <= base_iou <= max_iou <= 1")
        if self.density_beta < 0:
            raise ValueError("density_beta must be non-negative")
        if len(self.quality_weights) != 3 or any(weight < 0 for weight in self.quality_weights):
            raise ValueError("quality_weights must contain three non-negative values")
        if abs(sum(self.quality_weights) - 1.0) > 1e-6:
            raise ValueError("quality_weights must sum to one")
        if self.ideal_aspect_ratio <= 0 or self.max_aspect_deviation <= 1:
            raise ValueError("Aspect-ratio parameters are invalid")
        if self.density_chunk_size <= 0:
            raise ValueError("density_chunk_size must be positive")


def _density_and_agreement(
    boxes: Tensor,
    class_ids: Tensor,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Compute normalized local density and maximum same-class IoU.

    Density follows equation (19), then is normalized by the number of possible
    same-class neighbors to obtain ``D_i`` in ``[0, 1]``.
    """
    count = boxes.shape[0]
    density_sum = boxes.new_zeros(count)
    max_overlap = boxes.new_zeros(count)
    same_class_count = boxes.new_zeros(count)

    all_indices = torch.arange(count, device=boxes.device)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        overlaps = box_iou(boxes[start:end], boxes)
        same_class = class_ids[start:end, None].eq(class_ids[None, :])
        self_mask = all_indices[start:end, None].eq(all_indices[None, :])
        valid = same_class & ~self_mask
        masked = overlaps * valid.to(overlaps.dtype)
        density_sum[start:end] = masked.sum(dim=1)
        max_overlap[start:end] = masked.max(dim=1).values
        same_class_count[start:end] = valid.sum(dim=1).to(overlaps.dtype)

    normalized_density = density_sum / same_class_count.clamp_min(1.0)
    return normalized_density.clamp(0.0, 1.0), max_overlap.clamp(0.0, 1.0)


def density_adaptive_thresholds(density: Tensor, config: QHANMSConfig) -> Tensor:
    """Equation (20): ``T_i = clamp(T_base - beta * D_i)``."""
    return (config.base_iou - config.density_beta * density).clamp(
        config.min_iou, config.max_iou
    )


def boundary_quality(boxes: Tensor, config: QHANMSConfig) -> Tensor:
    """Morphological quality based on deviation from the ideal aspect ratio.

    The absolute logarithmic deviation is used to keep the score symmetric for
    reciprocal aspect ratios and bounded in ``[0, 1]``.
    """
    width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-7)
    height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-7)
    aspect_ratio = width / height
    deviation = torch.abs(torch.log(aspect_ratio / config.ideal_aspect_ratio))
    denominator = torch.log(
        torch.tensor(config.max_aspect_deviation, device=boxes.device, dtype=boxes.dtype)
    )
    score = config.boundary_lambda * (1.0 - deviation / denominator)
    return score.clamp(0.0, 1.0)


def _pair_threshold(
    selected_threshold: Tensor,
    candidate_thresholds: Tensor,
    rule: str,
) -> Tensor:
    if rule == "candidate":
        return candidate_thresholds
    if rule == "selected":
        return selected_threshold.expand_as(candidate_thresholds)
    if rule == "min":
        return torch.minimum(selected_threshold, candidate_thresholds)
    if rule == "mean":
        return 0.5 * (selected_threshold + candidate_thresholds)
    raise ValueError(f"Unknown threshold rule: {rule}")


def heterogeneous_greedy_nms(
    boxes: Tensor,
    ranking_scores: Tensor,
    class_ids: Tensor,
    thresholds: Tensor,
    *,
    max_det: int,
    threshold_rule: str,
) -> Tensor:
    """Greedy class-aware NMS with a different IoU threshold per box."""
    order = torch.argsort(ranking_scores, descending=True)
    kept: list[Tensor] = []

    while order.numel() > 0 and len(kept) < max_det:
        selected = order[0]
        kept.append(selected)
        if order.numel() == 1:
            break

        candidates = order[1:]
        overlaps = box_iou(boxes[selected : selected + 1], boxes[candidates]).squeeze(0)
        same_class = class_ids[candidates].eq(class_ids[selected])
        pair_threshold = _pair_threshold(
            thresholds[selected], thresholds[candidates], threshold_rule
        )
        suppress = same_class & (overlaps > pair_threshold)
        order = candidates[~suppress]

    if not kept:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return torch.stack(kept)


def qha_nms_boxes(
    boxes: Tensor,
    confidence_scores: Tensor,
    class_ids: Optional[Tensor] = None,
    *,
    localization_quality: Optional[Tensor] = None,
    config: Optional[QHANMSConfig] = None,
    max_det: int = 300,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the paper's two sequential QHA-NMS stages on decoded boxes.

    Args:
        boxes: ``[N, 4]`` boxes in ``xyxy`` format.
        confidence_scores: ``[N]`` classification confidence.
        class_ids: ``[N]`` class IDs. Defaults to one class.
        localization_quality: Optional IoU-aware score from a model quality
            head. This is the closest implementation of ``Q_IoU``. When it is
            unavailable, a documented proxy is used according to ``config``.
        config: QHA-NMS parameters.
        max_det: Maximum final detections.

    Returns:
        ``(keep_indices, final_quality_scores, inherited_thresholds)`` where
        scores and thresholds correspond to the original input boxes.
    """
    config = config or QHANMSConfig()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    if confidence_scores.shape != (boxes.shape[0],):
        raise ValueError("confidence_scores must have shape [N]")
    if class_ids is None:
        class_ids = torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
    else:
        class_ids = class_ids.to(device=boxes.device, dtype=torch.long)
    if class_ids.shape != (boxes.shape[0],):
        raise ValueError("class_ids must have shape [N]")
    if boxes.shape[0] == 0:
        empty_idx = torch.empty(0, dtype=torch.long, device=boxes.device)
        empty = boxes.new_empty(0)
        return empty_idx, empty, empty

    confidence_scores = confidence_scores.clamp(0.0, 1.0)
    density, agreement = _density_and_agreement(
        boxes, class_ids, config.density_chunk_size
    )
    thresholds = density_adaptive_thresholds(density, config)

    # Stage I: density-aware preliminary suppression using confidence ranking.
    stage1_keep = heterogeneous_greedy_nms(
        boxes,
        confidence_scores,
        class_ids,
        thresholds,
        max_det=boxes.shape[0],
        threshold_rule=config.threshold_rule,
    )

    stage1_boxes = boxes[stage1_keep]
    stage1_conf = confidence_scores[stage1_keep]
    stage1_classes = class_ids[stage1_keep]
    stage1_thresholds = thresholds[stage1_keep]

    if localization_quality is not None:
        localization_quality = localization_quality.to(
            device=boxes.device, dtype=boxes.dtype
        ).clamp(0.0, 1.0)
        if localization_quality.shape != (boxes.shape[0],):
            raise ValueError("localization_quality must have shape [N]")
        q_iou = localization_quality[stage1_keep]
    elif config.localization_quality_mode == "agreement":
        # Engineering fallback for a standard YOLO head. It is not a learned
        # IoU prediction. Isolated boxes fall back to confidence rather than
        # being assigned zero localization quality.
        stage1_agreement = agreement[stage1_keep]
        q_iou = torch.where(stage1_agreement > 0, stage1_agreement, stage1_conf)
    else:
        q_iou = stage1_conf

    q_bound = boundary_quality(stage1_boxes, config)
    w_conf, w_iou, w_bound = config.quality_weights
    stage1_quality = (
        w_conf * stage1_conf + w_iou * q_iou + w_bound * q_bound
    ).clamp(0.0, 1.0)

    # Stage II: quality-aware ranking using thresholds inherited from Stage I.
    stage2_local_keep = heterogeneous_greedy_nms(
        stage1_boxes,
        stage1_quality,
        stage1_classes,
        stage1_thresholds,
        max_det=max_det,
        threshold_rule=config.threshold_rule,
    )
    final_keep = stage1_keep[stage2_local_keep]

    full_quality = boxes.new_zeros(boxes.shape[0])
    full_quality[stage1_keep] = stage1_quality
    return final_keep, full_quality, thresholds


def qha_non_max_suppression(
    prediction: Tensor | Sequence[Tensor],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[Iterable[int]] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: Sequence[Tensor] = (),
    max_det: int = 300,
    nc: int = 0,
    max_time_img: float = 0.05,
    max_nms: int = 3000,
    max_wh: int = 7680,
    *,
    qha_config: Optional[QHANMSConfig] = None,
    replace_output_confidence: bool = False,
    **_: object,
) -> list[Tensor]:
    """Ultralytics-compatible wrapper around :func:`qha_nms_boxes`.

    Expected prediction layout is the legacy YOLOv8 format
    ``[batch, 4 + num_classes + num_masks, anchors]``. ``[batch, anchors,
    channels]`` is also accepted and normalized automatically.
    """
    del max_wh  # Class-aware handling is performed explicitly.
    if not 0 <= conf_thres <= 1 or not 0 <= iou_thres <= 1:
        raise ValueError("conf_thres and iou_thres must be in [0, 1]")
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if prediction.ndim != 3:
        raise ValueError(f"prediction must be rank 3, got {tuple(prediction.shape)}")

    # Normalize to [B, C, N].
    if prediction.shape[1] > prediction.shape[2]:
        prediction = prediction.transpose(1, 2)

    device = prediction.device
    original_device = device
    if device.type == "mps":
        prediction = prediction.cpu()
        device = prediction.device

    batch_size, channels, _ = prediction.shape
    nc = nc or channels - 4
    if nc <= 0 or nc > channels - 4:
        raise ValueError(f"Invalid nc={nc} for prediction with {channels} channels")
    nm = channels - nc - 4
    class_end = 4 + nc
    candidate_mask = prediction[:, 4:class_end].amax(dim=1) > conf_thres

    base_config = qha_config or QHANMSConfig(base_iou=iou_thres)
    if qha_config is not None and abs(qha_config.base_iou - iou_thres) > 1e-12:
        # Explicit config is authoritative.
        base_config = qha_config

    allowed_classes = None
    if classes is not None:
        allowed_classes = torch.as_tensor(list(classes), device=device, dtype=torch.long)

    started = time.time()
    outputs = [prediction.new_zeros((0, 6 + nm)) for _ in range(batch_size)]

    for image_index, raw in enumerate(prediction):
        raw = raw.transpose(0, 1)[candidate_mask[image_index]]
        if labels and image_index < len(labels) and len(labels[image_index]):
            label = labels[image_index]
            injected = raw.new_zeros((len(label), channels))
            injected[:, :4] = label[:, 1:5]
            injected[torch.arange(len(label), device=device), label[:, 0].long() + 4] = 1.0
            raw = torch.cat((raw, injected), dim=0)
        if raw.numel() == 0:
            continue

        boxes_xyxy = xywh2xyxy(raw[:, :4])
        class_scores = raw[:, 4:class_end]
        masks = raw[:, class_end:]

        if multi_label and nc > 1:
            row_index, class_index = (class_scores > conf_thres).nonzero(as_tuple=True)
            boxes = boxes_xyxy[row_index]
            scores = class_scores[row_index, class_index]
            class_ids = class_index
            masks = masks[row_index]
        else:
            scores, class_ids = class_scores.max(dim=1)
            valid = scores > conf_thres
            boxes = boxes_xyxy[valid]
            scores = scores[valid]
            class_ids = class_ids[valid]
            masks = masks[valid]

        if allowed_classes is not None and boxes.numel():
            valid = (class_ids[:, None] == allowed_classes[None, :]).any(dim=1)
            boxes, scores, class_ids, masks = (
                boxes[valid],
                scores[valid],
                class_ids[valid],
                masks[valid],
            )
        if boxes.numel() == 0:
            continue

        # Limit the quadratic density computation to the strongest candidates.
        order = torch.argsort(scores, descending=True)[:max_nms]
        boxes, scores, class_ids, masks = (
            boxes[order],
            scores[order],
            class_ids[order],
            masks[order],
        )
        nms_classes = torch.zeros_like(class_ids) if agnostic else class_ids
        keep, quality, _ = qha_nms_boxes(
            boxes,
            scores,
            nms_classes,
            config=base_config,
            max_det=max_det,
        )
        selected_scores = quality[keep] if replace_output_confidence else scores[keep]
        outputs[image_index] = torch.cat(
            (
                boxes[keep],
                selected_scores[:, None],
                class_ids[keep, None].to(boxes.dtype),
                masks[keep],
            ),
            dim=1,
        )
        if original_device.type == "mps":
            outputs[image_index] = outputs[image_index].to(original_device)
        if time.time() - started > 0.5 + max_time_img * batch_size:
            break

    return outputs
