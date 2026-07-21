# Integration Guide for the Original YOLOv8 Fork

The uploaded project uses the older module layout `ultralytics.yolo.*`. The files in `daq_yolo/integration/legacy_yolov8.py` target that layout.

## 1. Make `D_EMA` visible to the YAML parser

In the file containing `parse_model` (normally `ultralytics/nn/tasks.py`), import the module:

```python
from daq_yolo.modules.d_ema import D_EMA
```

Then add this branch inside `parse_model`, before the generic `else` branch:

```python
elif m is D_EMA:
    c1 = ch[f]
    args = [c1, *args]  # YAML args are [G_max, T_c]
    c2 = c1             # attention block preserves channels
```

This is essential. Do not hard-code `channels=64/128/256` in the YAML because YOLO width scaling changes the actual channels.

## 2. Use the provided model YAML

```text
configs/yolov8n-daq.yaml
```

It inserts D-EMA after the P3 and P5 neck features, consistent with the manuscript architecture figure. Adjust the locations only when your own ablation configuration requires it.

## 3. Activate AQFL in the actual loss path

Use `DAQDetectionTrainer` from `legacy_yolov8.py`, or copy the following core logic into your existing `Loss.__call__` after assignment:

```python
aligned_iou = aligned_box_iou(pred_bboxes, target_bboxes)
aligned_iou = aligned_iou * fg_mask.to(aligned_iou.dtype)

loss_cls = aqfl(
    pred_scores,
    target_scores.to(pred_scores.dtype),
    iou_scores=aligned_iou,
    positive_mask=fg_mask,
)
loss[1] = loss_cls.sum() / target_scores.sum().clamp_min(1.0)
```

Remove or disable the previous classification line:

```python
loss[1] = BCEWithLogitsLoss(...)
```

## 4. Install QHA-NMS once for validation/inference

```python
from daq_yolo.integration.legacy_yolov8 import QHANMSPatch
from daq_yolo.nms import QHANMSConfig

config = QHANMSConfig(base_iou=0.45, density_beta=0.20)
with QHANMSPatch(config):
    trainer.train()
```

The patch affects validation and inference calls that resolve `ops.non_max_suppression` at runtime.

## 5. Do not run “NMS stage 1 training” and “NMS stage 2 training” as two unrelated models

NMS is a post-processing operator. In this implementation, the two stages execute sequentially for each image:

```text
candidate boxes
    -> density-aware heterogeneous suppression
    -> quality-aware ranking and inherited-threshold suppression
    -> final detections
```

If you later add a learned IoU-quality head, train that head with a differentiable quality target and pass its output to `qha_nms_boxes(localization_quality=...)`. The discrete NMS loop itself is not optimized by ordinary backpropagation.

## 6. Recommended first validation

```bash
pytest
python train_daq.py --model configs/yolov8n-daq.yaml --data /path/to/seed967.yaml --epochs 1
```

Check that:

- D-EMA output channels equal input channels.
- `cls_loss` comes from `AQFocalLossWithLogits`, not BCE.
- validation calls `qha_non_max_suppression`.
- no layer is created inside `forward()`.
- results are compared at identical confidence and IoU settings.
