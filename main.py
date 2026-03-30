# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .scripts.config import DASBandConfig
from .scripts.extract_tdms import extract_name_to_csv
from .scripts.io import ensure_dir, resolve_audio_path, save_json
from .scripts.pipeline import prepare_training_labels, run_inference, train_from_prep


def build_parser():
    parser = argparse.ArgumentParser(
        description="End-to-end DASBand workflow: TDMS -> DAS CSV -> pseudo band labels -> training -> output."
    )
    parser.add_argument("name", help="Sample name. Expected Airtag CSV: <data_root>/Airtag/<name>.csv")

    parser.add_argument("--data_root", default="Data", help="Data root directory.")
    parser.add_argument("--audio_dir", default=None, help="Audio directory override. Default: <data_root>/Audio")
    parser.add_argument("--airtag_dir", default=None, help="Airtag directory override. Default: <data_root>/Airtag")
    parser.add_argument("--das_dir", default=None, help="TDMS directory override. Default: <data_root>/DAS")
    parser.add_argument("--output_root", default="output", help="Output root directory.")

    parser.add_argument("--candidate_csv", default=None, help="Candidate point CSV. If omitted, derive from audio + peak channel.")
    parser.add_argument("--checkpoint", default=None, help="Optional pretrained checkpoint. If provided, skip training and only infer.")

    parser.add_argument("--skip_extract", action="store_true", help="Skip TDMS extraction and reuse existing CSV if present.")
    parser.add_argument("--overwrite_extract", action="store_true", help="Overwrite extracted DAS CSV when it already exists.")
    parser.add_argument("--dry_run_extract", action="store_true", help="Only print extraction plan and stop.")

    parser.add_argument("--trim_start_s", type=float, default=50.0)
    parser.add_argument("--trim_end_s", type=float, default=None)
    parser.add_argument("--das_fs", type=int, default=2000)
    parser.add_argument("--csv_utc_offset_hours", type=float, default=8.0)
    parser.add_argument("--skip_channels", type=int, default=18)
    parser.add_argument("--das_filter_method", choices=["filtfilt", "sosfilt"], default="sosfilt")

    parser.add_argument("--gaussian_sigma_ch", type=float, default=3.0)
    parser.add_argument("--label_mode", choices=["gaussian", "hard"], default="gaussian")
    parser.add_argument("--hard_band_radius_ch", type=int, default=3)
    parser.add_argument("--disable_signal_prior", action="store_true")
    parser.add_argument("--clean_outlier_threshold_ch", type=float, default=5.0)
    parser.add_argument("--clean_project_to_line", action="store_true")
    parser.add_argument("--clean_min_segment_points", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patch_frames", type=int, default=256)
    parser.add_argument("--patch_stride", type=int, default=128)
    parser.add_argument("--model_channels", type=int, default=32)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--loss_mask_weight", type=float, default=1.0)
    parser.add_argument("--loss_center_weight", type=float, default=0.3)
    parser.add_argument("--loss_smooth_weight", type=float, default=0.2)
    parser.add_argument("--loss_tv_weight", type=float, default=0.1)
    parser.add_argument("--loss_area_weight", type=float, default=0.01)

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

    parser.add_argument("--stop_after_prepare", action="store_true", help="Stop after pseudo label preparation.")
    parser.add_argument("--stop_after_train", action="store_true", help="Stop after training and do not run inference.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _resolve_paths(args):
    data_root = Path(args.data_root).expanduser().resolve()
    audio_dir = Path(args.audio_dir).expanduser().resolve() if args.audio_dir else (data_root / "Audio").resolve()
    airtag_dir = Path(args.airtag_dir).expanduser().resolve() if args.airtag_dir else (data_root / "Airtag").resolve()
    das_dir = Path(args.das_dir).expanduser().resolve() if args.das_dir else (data_root / "DAS").resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    sample_root = output_root / args.name.lower()
    return data_root, audio_dir, airtag_dir, das_dir, output_root, sample_root


def _find_audio_path(audio_dir: Path, data_root: Path, name: str) -> Optional[str]:
    for ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".MP4", ".mp4", ".MOV", ".mov"]:
        cand = audio_dir / f"{name}{ext}"
        if cand.exists():
            return str(cand.resolve())
    cfg_for_audio = DASBandConfig(name=name, data_root=str(data_root))
    resolved_audio = resolve_audio_path(cfg_for_audio)
    return str(resolved_audio) if resolved_audio is not None else None


def _build_config(args, das_csv: str, audio_path: Optional[str]):
    return DASBandConfig(
        name=args.name.lower(),
        data_root=args.data_root,
        output_root=args.output_root,
        das_csv=das_csv,
        audio_path=audio_path,
        candidate_csv=args.candidate_csv,
        das_fs=args.das_fs,
        trim_start_s=args.trim_start_s,
        trim_end_s=args.trim_end_s,
        csv_utc_offset_hours=args.csv_utc_offset_hours,
        skip_channels=args.skip_channels,
        das_filter_method=args.das_filter_method,
        gaussian_sigma_ch=args.gaussian_sigma_ch,
        label_mode=args.label_mode,
        hard_band_radius_ch=args.hard_band_radius_ch,
        use_signal_prior=not args.disable_signal_prior,
        clean_outlier_threshold_ch=args.clean_outlier_threshold_ch,
        clean_project_to_line=args.clean_project_to_line,
        clean_min_segment_points=args.clean_min_segment_points,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patch_frames=args.patch_frames,
        patch_stride=args.patch_stride,
        model_channels=args.model_channels,
        model_dropout=args.model_dropout,
        device=args.device,
        loss_mask_weight=args.loss_mask_weight,
        loss_center_weight=args.loss_center_weight,
        loss_smooth_weight=args.loss_smooth_weight,
        loss_tv_weight=args.loss_tv_weight,
        loss_area_weight=args.loss_area_weight,
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
        seed=args.seed,
    )


def main():
    args = build_parser().parse_args()
    args.name = args.name.lower()

    data_root, audio_dir, airtag_dir, das_dir, output_root, sample_root = _resolve_paths(args)
    signals_dir = ensure_dir(sample_root / "signals")
    workflow_dir = ensure_dir(sample_root / "dasband" / "workflow")

    das_csv_path = signals_dir / f"{args.name}.csv"
    if args.skip_extract:
        if not das_csv_path.exists():
            raise FileNotFoundError(f"--skip_extract was set but CSV does not exist: {das_csv_path}")
        extract_info = {"csv_path": str(das_csv_path), "rows": None, "channels": None}
        print(f"[Workflow] reuse extracted CSV: {das_csv_path}")
    else:
        extract_info = extract_name_to_csv(
            name=args.name,
            airtag_csv_dir=str(airtag_dir),
            tdms_dir=str(das_dir),
            output_dir=str(signals_dir),
            fs=float(args.das_fs),
            csv_utc_offset_hours=float(args.csv_utc_offset_hours),
            skip_channels=int(args.skip_channels),
            overwrite=bool(args.overwrite_extract),
            dry_run=bool(args.dry_run_extract),
        )
        print(f"[Workflow] extracted CSV: {extract_info['csv_path']}")
        if args.dry_run_extract:
            return

    audio_path = None
    if args.candidate_csv is None:
        audio_path = _find_audio_path(audio_dir, data_root, args.name)

    config = _build_config(args, str(das_csv_path), audio_path)

    summary = {
        "name": args.name,
        "data_root": str(data_root),
        "audio_dir": str(audio_dir),
        "airtag_dir": str(airtag_dir),
        "das_dir": str(das_dir),
        "output_root": str(output_root),
        "signals_csv": str(das_csv_path),
        "candidate_csv": args.candidate_csv,
        "audio_path": audio_path,
        "checkpoint": args.checkpoint,
        "extract_info": {
            "csv_path": str(extract_info["csv_path"]),
            "rows": extract_info.get("rows"),
            "channels": extract_info.get("channels"),
        },
        "config": config.to_dict(),
    }
    save_json(workflow_dir / "workflow_args.json", summary)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        prep_result = prepare_training_labels(config)
        print(f"[Workflow] prepare dir: {prep_result['prep_dir']}")
        if args.stop_after_prepare:
            return

        train_result = train_from_prep(str(prep_result["prep_dir"]), config)
        checkpoint_path = str(train_result["checkpoint"])
        print(f"[Workflow] trained checkpoint: {checkpoint_path}")
        if args.stop_after_train:
            return
    else:
        print(f"[Workflow] skip training, use checkpoint: {checkpoint_path}")

    infer_result = run_inference(str(das_csv_path), str(checkpoint_path), config)
    print(f"[Workflow] inference dir: {infer_result['output_dir']}")


if __name__ == "__main__":
    main()
