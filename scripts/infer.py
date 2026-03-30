# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from .config import DASBandConfig
from .pipeline import run_inference


def build_parser():
    parser = argparse.ArgumentParser(description="Infer dasband mask and decode the main track.")
    parser.add_argument("--das_csv", required=True, help="Path to DAS CSV.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained dasband_model.pt.")
    parser.add_argument("--name", default=None, help="Sample name for output organization.")
    parser.add_argument("--output_root", default="output", help="Output root.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--das_fs", type=int, default=2000)
    parser.add_argument("--trim_start_s", type=float, default=50.0)
    parser.add_argument("--trim_end_s", type=float, default=None)
    parser.add_argument("--patch_frames", type=int, default=256)
    parser.add_argument("--patch_stride", type=int, default=128)
    parser.add_argument("--dp_jump_penalty", type=float, default=0.8)
    parser.add_argument("--dp_curvature_penalty", type=float, default=0.2)
    parser.add_argument("--dp_max_jump_ch", type=int, default=6)
    parser.add_argument("--centroid_threshold", type=float, default=0.2)
    parser.add_argument("--decode_mode", choices=["kalman", "dp"], default="kalman")
    parser.add_argument("--kalman_process_var", type=float, default=2.0)
    parser.add_argument("--kalman_measurement_var", type=float, default=0.8)
    parser.add_argument("--kalman_measurement_var_floor", type=float, default=0.15)
    parser.add_argument("--sigma_scale", type=float, default=0.7)
    parser.add_argument("--sigma_min", type=float, default=0.5)
    parser.add_argument("--sigma_max", type=float, default=3.0)
    return parser


def main():
    args = build_parser().parse_args()
    config = DASBandConfig(
        name=args.name,
        output_root=args.output_root,
        das_fs=args.das_fs,
        trim_start_s=args.trim_start_s,
        trim_end_s=args.trim_end_s,
        device=args.device,
        patch_frames=args.patch_frames,
        patch_stride=args.patch_stride,
        dp_jump_penalty=args.dp_jump_penalty,
        dp_curvature_penalty=args.dp_curvature_penalty,
        dp_max_jump_ch=args.dp_max_jump_ch,
        centroid_threshold=args.centroid_threshold,
        decode_mode=args.decode_mode,
        kalman_process_var=args.kalman_process_var,
        kalman_measurement_var=args.kalman_measurement_var,
        kalman_measurement_var_floor=args.kalman_measurement_var_floor,
        sigma_scale=args.sigma_scale,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    result = run_inference(args.das_csv, args.checkpoint, config)
    print(f"[Infer] saved to: {Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
