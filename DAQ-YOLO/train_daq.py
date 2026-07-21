"""Train DAQ-YOLO with the legacy Ultralytics API used by the original code."""

from __future__ import annotations

import argparse

from daq_yolo.integration.legacy_yolov8 import DAQDetectionTrainer, QHANMSPatch
from daq_yolo.nms.qha_nms import QHANMSConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to yolov8n-daq.yaml")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="daq_yolov8n")
    parser.add_argument("--aqfl-alpha", type=float, default=0.25)
    parser.add_argument("--aqfl-gamma0", type=float, default=2.0)
    parser.add_argument("--aqfl-lambda", type=float, default=1.0)
    parser.add_argument("--aqfl-beta", type=float, default=1.0)
    parser.add_argument("--nms-base-iou", type=float, default=0.45)
    parser.add_argument("--nms-density-beta", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "model": args.model,
        "data": args.data,
        "device": args.device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "name": args.name,
        "optimizer": "AdamW",
        "lr0": 1e-2,
        "weight_decay": 5e-4,
        "warmup_epochs": 3.0,
        "patience": 50,
        "aqfl_alpha": args.aqfl_alpha,
        "aqfl_gamma0": args.aqfl_gamma0,
        "aqfl_lambda": args.aqfl_lambda,
        "aqfl_beta": args.aqfl_beta,
        "aqfl_formulation": "paper",
        "aqfl_detach_iou": False,
    }
    nms_config = QHANMSConfig(
        base_iou=args.nms_base_iou,
        density_beta=args.nms_density_beta,
    )
    # QHA-NMS affects validation/inference. AQFL and D-EMA are trained normally;
    # NMS itself is post-processing and does not require a second 200-epoch run.
    with QHANMSPatch(nms_config):
        trainer = DAQDetectionTrainer(overrides=overrides)
        trainer.train()


if __name__ == "__main__":
    main()
