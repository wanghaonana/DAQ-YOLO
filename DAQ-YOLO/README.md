# DAQ-YOLO: Paper-aligned Core Components

This repository contains a cleaned and open-source-ready implementation of the three core components described in the DAQ-YOLO manuscript:

- **D-EMA** — Dynamic Efficient Multi-scale Attention
- **AQFL** — Adaptive Quality-aware Focal Loss
- **QHA-NMS** — Quality-Augmented Heterogeneous Adaptive NMS

The code was reorganized from an experimental YOLOv8 project and corrected so the active training/inference path matches the manuscript equations more closely.

## Project structure

```text
DAQ-YOLO-open-source/
├── daq_yolo/
│   ├── losses/aqfl.py                 # Equations (12)-(18)
│   ├── modules/d_ema.py               # Equations (1)-(11)
│   ├── nms/qha_nms.py                 # Equations (19)-(24)
│   └── integration/legacy_yolov8.py   # Hooks for your YOLOv8 fork
├── configs/yolov8n-daq.yaml
├── docs/
│   ├── ALGORITHM_ALIGNMENT.md
│   └── INTEGRATION_GUIDE.md
├── images/
│   ├── uploads/
│   ├── architecture/
│   ├── results/
│   └── samples/
├── IMAGE_UPLOAD_GUIDE_CN.md
├── tests/
├── train_daq.py
├── requirements.txt
└── pyproject.toml
```


## 中文快速开始：上传约 100 张图片

代码包已经预建图片目录。最简单的做法是先上传全部代码，再进入 `images/uploads/`，分批拖入图片并提交。

```text
images/
├── uploads/          # 普通项目图片，推荐直接上传到这里
├── architecture/     # 网络结构图、流程图
├── results/          # 检测结果、对比图
└── samples/          # 少量示例输入图
```

完整中文步骤见 [`IMAGE_UPLOAD_GUIDE_CN.md`](IMAGE_UPLOAD_GUIDE_CN.md)。

在 README 中显示图片：

```markdown
![检测结果](images/uploads/result-001.png)
```

## What was corrected

### AQFL

The original training file instantiated `BCEWithLogitsLoss`, so AQFL was not active. The new `DAQDetectionLoss` computes aligned predicted-box/assigned-box IoU and passes it into AQFL.

The default implementation uses the manuscript expression:

```text
L = -alpha_t * (1-q)^beta * (1-p_t)^gamma_i * log(p_t)
```

For reproducibility with older experiments, `formulation="soft_bce"` remains available, but it differs from `-log(p_t)` for continuous labels.

### D-EMA

The implementation now performs:

1. Dynamic channel grouping.
2. Horizontal/vertical spatial modeling.
3. A local `3x3 Conv + BN + ReLU` branch.
4. `F_ref = F_local * A_spatial`.
5. `F_enh = F_ref + alpha * x`.

No convolution is created inside `forward()`, and the input/output channel count remains unchanged.

### QHA-NMS

QHA-NMS is now one **sequential two-stage post-processing function**, not two independent 200-epoch training runs:

1. Stage I computes local density and a per-box threshold `T_i`.
2. Stage II ranks Stage-I candidates by confidence, localization quality, and boundary quality, then performs heterogeneous NMS using the inherited thresholds.

`torchvision.ops.nms` cannot implement per-box thresholds, so the repository includes a custom pure-PyTorch greedy NMS.

## Installation

```bash
pip install -e .
pip install -r requirements.txt
pytest
```

The integration file targets the legacy import layout used by the original project:

```python
from ultralytics.yolo.utils import ops
```

Use the same YOLOv8 fork/version as your original training code. See [Integration Guide](docs/INTEGRATION_GUIDE.md).

## Training

```bash
python train_daq.py \
  --model configs/yolov8n-daq.yaml \
  --data /path/to/seed967.yaml \
  --device 0 \
  --epochs 200 \
  --imgsz 640 \
  --batch 16
```

Before training, add the small `parse_model` branch shown in the integration guide so the current input-channel count is injected into `D_EMA`.

## Standalone use

### AQFL

```python
from daq_yolo.losses import AQFocalLossWithLogits

criterion = AQFocalLossWithLogits(
    alpha=0.25,
    gamma0=2.0,
    lambda_=1.0,
    beta=1.0,
    formulation="paper",
)
loss = criterion(logits, soft_targets, iou_scores=aligned_iou)
```

### D-EMA

```python
from daq_yolo.modules import D_EMA

attention = D_EMA(channels=64, G_max=8, T_c=4)
y = attention(x)
```

### QHA-NMS

```python
from daq_yolo.nms import QHANMSConfig, qha_nms_boxes

config = QHANMSConfig(
    base_iou=0.45,
    min_iou=0.30,
    max_iou=0.70,
    density_beta=0.20,
    quality_weights=(0.60, 0.30, 0.10),
    ideal_aspect_ratio=1.0,
)
keep, quality, thresholds = qha_nms_boxes(
    boxes_xyxy,
    confidence,
    class_ids,
    localization_quality=predicted_iou,  # optional IoU-aware head output
    config=config,
)
```

## Important note about `Q_IoU`

A standard YOLOv8 detection head does not output a separate localization-quality score. The most faithful option is to add an IoU-aware quality head and pass its output as `localization_quality`. When that output is absent, this repository uses a clearly documented same-class box-agreement proxy. The fallback is practical, but it should not be described as a learned Alpha-IoU quality prediction.

## Reproducibility

Record these values in every experiment:

- `alpha`, `gamma0`, `lambda`, and `beta` for AQFL.
- `G_max`, `T_c`, and `alpha_init` for D-EMA.
- `T_min`, `T_max`, `T_base`, density coefficient, quality weights, ideal aspect ratio, and `Q_IoU` source for QHA-NMS.
- Ultralytics commit/version and the exact model YAML.

## License

This repository is distributed under **GPL-3.0-or-later** because the integration code is derived from a GPL-licensed Ultralytics YOLOv8 training path. See `LICENSE` and `NOTICE.md`.
