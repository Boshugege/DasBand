# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from .config import DASBandConfig
from .pipeline import train_from_prep


def build_parser():
    parser = argparse.ArgumentParser(description="Train dasband 2D mask model from prepared pseudo labels.")
    parser.add_argument("--prep_dir", required=True, help="Prepared label directory from dasband.scripts.prepare_labels.")
    parser.add_argument("--name", default=None, help="Sample name for output organization.")
    parser.add_argument("--output_root", default="output", help="Output root.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patch_frames", type=int, default=256)
    parser.add_argument("--patch_stride", type=int, default=128)
    parser.add_argument("--model_channels", type=int, default=32)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--loss_mask_weight", type=float, default=1.0)
    parser.add_argument("--loss_center_weight", type=float, default=0.3)
    parser.add_argument("--loss_smooth_weight", type=float, default=0.2)
    parser.add_argument("--loss_tv_weight", type=float, default=0.1)
    parser.add_argument("--loss_area_weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    config = DASBandConfig(
        name=args.name,
        output_root=args.output_root,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patch_frames=args.patch_frames,
        patch_stride=args.patch_stride,
        model_channels=args.model_channels,
        model_dropout=args.model_dropout,
        loss_mask_weight=args.loss_mask_weight,
        loss_center_weight=args.loss_center_weight,
        loss_smooth_weight=args.loss_smooth_weight,
        loss_tv_weight=args.loss_tv_weight,
        loss_area_weight=args.loss_area_weight,
        seed=args.seed,
    )
    result = train_from_prep(args.prep_dir, config)
    print(f"[Train] checkpoint: {Path(result['checkpoint']).resolve()}")


if __name__ == "__main__":
    main()
