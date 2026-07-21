"""Adaptive Quality-aware Focal Loss (AQFL).

This module follows equations (12)-(18) in the DAQ-YOLO manuscript.  The
``paper`` formulation uses ``-log(p_t)`` exactly.  A ``soft_bce`` compatibility
mode is also supplied because many existing YOLO implementations use BCE for
continuous targets; it is not mathematically identical to the paper when
``target`` is strictly between zero and one.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def aligned_box_iou(pred_boxes: Tensor, target_boxes: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute element-wise IoU for aligned ``xyxy`` boxes.

    Args:
        pred_boxes: Tensor with shape ``(..., 4)``.
        target_boxes: Tensor with the same shape as ``pred_boxes``.
        eps: Numerical stability constant.

    Returns:
        IoU tensor with shape ``(...)``.
    """
    if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[-1] != 4:
        raise ValueError(
            "pred_boxes and target_boxes must have identical shape (..., 4), "
            f"got {tuple(pred_boxes.shape)} and {tuple(target_boxes.shape)}"
        )

    left_top = torch.maximum(pred_boxes[..., :2], target_boxes[..., :2])
    right_bottom = torch.minimum(pred_boxes[..., 2:], target_boxes[..., 2:])
    intersection_wh = (right_bottom - left_top).clamp_min(0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    pred_wh = (pred_boxes[..., 2:] - pred_boxes[..., :2]).clamp_min(0)
    target_wh = (target_boxes[..., 2:] - target_boxes[..., :2]).clamp_min(0)
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    target_area = target_wh[..., 0] * target_wh[..., 1]
    union = pred_area + target_area - intersection
    return intersection / union.clamp_min(eps)


class AQFocalLossWithLogits(nn.Module):
    r"""Adaptive Quality-aware Focal Loss operating on raw logits.

    Paper equations:

    .. math::
        p_t = yp + (1-y)(1-p)

    .. math::
        \alpha_t = y\alpha + (1-y)(1-\alpha)

    .. math::
        \gamma_i = \gamma_0 + \lambda(1-p_t)

    .. math::
        w_q = (1-q)^\beta

    .. math::
        L_i = -\alpha_t w_q (1-p_t)^{\gamma_i}\log(p_t)

    Args:
        alpha: Base class-balancing factor.
        gamma0: Base focusing parameter.
        lambda_: Sample-difficulty modulation coefficient.
        beta: IoU-quality modulation coefficient.
        reduction: ``none``, ``mean`` or ``sum``.
        formulation: ``paper`` for equation (17), or ``soft_bce`` for backward
            compatibility with BCE-based soft-label implementations.
        detach_iou: Detach IoU before weighting. Keep ``False`` for direct
            classification-localization coupling; use ``True`` if training is
            numerically unstable.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma0: float = 2.0,
        lambda_: float = 1.0,
        beta: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "none",
        formulation: Literal["paper", "soft_bce"] = "paper",
        detach_iou: bool = False,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if gamma0 <= 0.0 or lambda_ < 0.0 or beta < 0.0:
            raise ValueError("gamma0 must be > 0; lambda_ and beta must be >= 0")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        if formulation not in {"paper", "soft_bce"}:
            raise ValueError(f"Unsupported formulation: {formulation}")

        self.alpha = float(alpha)
        self.gamma0 = float(gamma0)
        self.lambda_ = float(lambda_)
        self.beta = float(beta)
        self.reduction = reduction
        self.formulation = formulation
        self.detach_iou = bool(detach_iou)
        self.eps = float(eps)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        iou_scores: Optional[Tensor] = None,
        pred_boxes: Optional[Tensor] = None,
        target_boxes: Optional[Tensor] = None,
        positive_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute AQFL.

        ``iou_scores`` may have one fewer dimension than ``logits`` (for
        example ``[B, A]`` versus ``[B, A, C]``); it is automatically expanded
        along the class dimension.
        """
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have identical shape, got "
                f"{tuple(logits.shape)} and {tuple(targets.shape)}"
            )

        targets = targets.to(dtype=logits.dtype)
        if iou_scores is None:
            if pred_boxes is None or target_boxes is None:
                raise ValueError(
                    "Pass iou_scores or both pred_boxes and target_boxes."
                )
            iou_scores = aligned_box_iou(pred_boxes, target_boxes, self.eps)

        iou_scores = iou_scores.to(device=logits.device, dtype=logits.dtype)
        while iou_scores.ndim < logits.ndim:
            iou_scores = iou_scores.unsqueeze(-1)
        try:
            iou_scores = torch.broadcast_to(iou_scores, logits.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"iou_scores shape {tuple(iou_scores.shape)} cannot broadcast "
                f"to logits shape {tuple(logits.shape)}"
            ) from exc

        if positive_mask is not None:
            mask = positive_mask.to(device=logits.device, dtype=torch.bool)
            while mask.ndim < logits.ndim:
                mask = mask.unsqueeze(-1)
            mask = torch.broadcast_to(mask, logits.shape)
            iou_scores = torch.where(mask, iou_scores, torch.zeros_like(iou_scores))

        if self.detach_iou:
            iou_scores = iou_scores.detach()
        q = iou_scores.clamp(0.0, 1.0)

        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        p_t = p_t.clamp(self.eps, 1.0 - self.eps)

        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        gamma_i = self.gamma0 + self.lambda_ * (1.0 - p_t)
        quality_weight = (1.0 - q).pow(self.beta)
        focal_weight = (1.0 - p_t).pow(gamma_i)

        if self.formulation == "paper":
            base_loss = -torch.log(p_t)
        else:
            base_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        loss = alpha_t * quality_weight * focal_weight * base_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
