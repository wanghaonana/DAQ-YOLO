import torch

from daq_yolo.nms.qha_nms import QHANMSConfig, qha_nms_boxes


def test_qha_nms_suppresses_duplicate_boxes() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.5, 0.5, 10.5, 10.5],
            [30.0, 30.0, 40.0, 40.0],
        ]
    )
    scores = torch.tensor([0.90, 0.80, 0.70])
    keep, quality, thresholds = qha_nms_boxes(boxes, scores, max_det=10)
    assert keep.tolist() == [0, 2]
    assert quality.shape == scores.shape
    assert thresholds.shape == scores.shape


def test_class_aware_nms_keeps_overlapping_different_classes() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.9, 0.8])
    classes = torch.tensor([0, 1])
    keep, _, _ = qha_nms_boxes(boxes, scores, classes, max_det=10)
    assert set(keep.tolist()) == {0, 1}


def test_high_density_lowers_threshold() -> None:
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    config = QHANMSConfig(base_iou=0.5, density_beta=0.2)
    _, _, thresholds = qha_nms_boxes(boxes, scores, config=config)
    assert thresholds[0] < thresholds[2]
