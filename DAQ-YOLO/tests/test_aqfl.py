import torch

from daq_yolo.losses.aqfl import AQFocalLossWithLogits, aligned_box_iou


def test_aligned_iou() -> None:
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
    targets = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]])
    result = aligned_box_iou(boxes, targets)
    assert torch.allclose(result, torch.tensor([1.0, 0.0]))


def test_binary_degenerates_to_focal_loss() -> None:
    logits = torch.tensor([[0.2, -0.4], [1.2, -2.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss_fn = AQFocalLossWithLogits(lambda_=0.0, beta=0.0, reduction="none")
    actual = loss_fn(logits, targets, iou_scores=torch.zeros(2, 1))

    probability = torch.sigmoid(logits)
    p_t = targets * probability + (1 - targets) * (1 - probability)
    alpha_t = targets * 0.25 + (1 - targets) * 0.75
    expected = -alpha_t * (1 - p_t).pow(2.0) * torch.log(p_t)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_aqfl_backward_is_finite() -> None:
    logits = torch.randn(2, 8, 1, requires_grad=True)
    targets = torch.rand(2, 8, 1)
    iou = torch.rand(2, 8)
    loss = AQFocalLossWithLogits(reduction="mean")(
        logits, targets, iou_scores=iou
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
