# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - import guard
    torch = None
    DataLoader = None

from .candidates import resolve_candidate_points
from .config import DASBandConfig
from .dataset import TimePatchDataset, build_patch_indices
from .decoder import (
    estimate_measurement_confidence,
    estimate_uncertainty,
    extract_path_dp,
    kalman_smooth_track,
    sigmoid,
    weighted_centroid,
)
from .io import (
    build_feature_cube,
    ensure_dir,
    load_das_csv,
    load_prepare_artifacts,
    save_json,
    save_prepare_artifacts,
    trim_das,
)
from .losses import compute_losses
from .model import DASBandUNet
from .pseudo_label import build_pseudo_label
from .trajectory_cleaning import fit_piecewise_trajectory
from .viz import plot_candidate_cleaning, plot_inference_result, plot_pseudo_label


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device_pref: str):
    if torch is None:
        raise RuntimeError("PyTorch is not installed. dasband training/inference requires torch.")
    pref = (device_pref or "auto").lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_training_labels(config: DASBandConfig, output_dir: Optional[str] = None):
    if not config.das_csv:
        raise ValueError("--das_csv is required for prepare_labels.")

    out_dir = Path(output_dir) if output_dir else config.resolve_run_root("prepare")
    ensure_dir(out_dir)

    das, ch_cols = load_das_csv(config.das_csv)
    das_trimmed, used_start_s, used_end_s = trim_das(das, config.das_fs, config.trim_start_s, config.trim_end_s)
    feature_cube, feature_names, frame_times, primary_energy = build_feature_cube(das_trimmed, config)

    raw_points_df = resolve_candidate_points(primary_energy, frame_times, config)
    clean_points_df, cleaning_summary, _ = fit_piecewise_trajectory(raw_points_df, config)
    pseudo_label, prior, centerline = build_pseudo_label(clean_points_df, frame_times, primary_energy, config)

    save_prepare_artifacts(
        out_dir,
        feature_cube=feature_cube,
        feature_names=feature_names,
        frame_times=frame_times,
        primary_energy=primary_energy,
        prior=prior,
        pseudo_label=pseudo_label,
        centerline=centerline,
        cleaned_points_df=clean_points_df,
        raw_points_df=raw_points_df,
        meta={
            "config": config.to_dict(),
            "used_trim_start_s": used_start_s,
            "used_trim_end_s": used_end_s,
            "num_channels": len(ch_cols),
            "feature_cube_shape": list(feature_cube.shape),
            "cleaning_summary": cleaning_summary,
        },
    )

    plot_candidate_cleaning(primary_energy, frame_times, raw_points_df, clean_points_df, str(out_dir / "candidate_cleaning.png"))
    plot_pseudo_label(primary_energy, frame_times, pseudo_label, centerline, str(out_dir / "pseudo_label.png"))

    return {
        "prep_dir": out_dir,
        "feature_cube": feature_cube,
        "feature_names": feature_names,
        "frame_times": frame_times,
        "primary_energy": primary_energy,
        "raw_points": raw_points_df,
        "clean_points": clean_points_df,
        "pseudo_label": pseudo_label,
        "prior": prior,
        "centerline": centerline,
        "cleaning_summary": cleaning_summary,
    }


def _build_model_and_device(feature_cube: np.ndarray, config: DASBandConfig):
    device = resolve_device(config.device)
    model = DASBandUNet(
        in_ch=int(feature_cube.shape[0]),
        base_ch=int(config.model_channels),
        dropout=float(config.model_dropout),
    ).to(device)
    return model, device


def train_from_prep(prep_dir: str, config: DASBandConfig, output_dir: Optional[str] = None):
    if torch is None:
        raise RuntimeError("PyTorch is not installed. dasband training requires torch.")

    set_seed(config.seed)
    prep = load_prepare_artifacts(prep_dir)
    out_dir = Path(output_dir) if output_dir else config.resolve_run_root("train")
    ensure_dir(out_dir)

    feature_cube = prep["feature_cube"].astype(np.float32)
    pseudo_label = prep["pseudo_label"].astype(np.float32)
    frame_times = prep["frame_times"].astype(np.float32)
    clean_points = prep["clean_points"]

    dataset = TimePatchDataset(
        feature_cube=feature_cube,
        pseudo_label=pseudo_label,
        frame_times=frame_times,
        points_df=clean_points,
        patch_frames=config.patch_frames,
        stride=config.patch_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    model, device = _build_model_and_device(feature_cube, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = bool(config.amp) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=use_amp)
    history = []

    model.train()
    for epoch in range(1, config.epochs + 1):
        running = {"total": 0.0, "mask": 0.0, "center": 0.0, "smooth": 0.0, "tv": 0.0, "area": 0.0}
        n_batches = 0
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            center_target = batch["center_target"].to(device)
            center_weight = batch["center_weight"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                losses = compute_losses(logits, y, center_target, center_weight, config)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()

            for key, value in losses.items():
                running[key] += float(value.detach().cpu().item())
            n_batches += 1

        epoch_row = {"epoch": epoch}
        for key in running:
            epoch_row[key] = running[key] / max(1, n_batches)
        history.append(epoch_row)
        print(
            f"[Train] epoch={epoch:03d} total={epoch_row['total']:.4f} mask={epoch_row['mask']:.4f} "
            f"center={epoch_row['center']:.4f} smooth={epoch_row['smooth']:.4f} tv={epoch_row['tv']:.4f}"
        )

    checkpoint_path = out_dir / "dasband_model.pt"
    prep_config_dict = prep["meta"].get("config", {})
    merged_config = DASBandConfig.from_dict({**prep_config_dict, **config.to_dict()}).to_dict()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": merged_config,
            "in_channels": int(feature_cube.shape[0]),
            "feature_names": prep["meta"].get("feature_names", []),
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    save_json(out_dir / "train_summary.json", {"checkpoint": str(checkpoint_path), "epochs": config.epochs})
    print(f"[Train] checkpoint saved to: {checkpoint_path}")
    return {"checkpoint": checkpoint_path, "history": history}


def load_model_checkpoint(checkpoint_path: str, device=None):
    if torch is None:
        raise RuntimeError("PyTorch is not installed. dasband inference requires torch.")
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
    cfg = DASBandConfig.from_dict(ckpt["config"])
    model = DASBandUNet(
        in_ch=int(ckpt["in_channels"]),
        base_ch=int(cfg.model_channels),
        dropout=float(cfg.model_dropout),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt


def infer_mask(feature_cube: np.ndarray, checkpoint_path: str, config_override: Optional[DASBandConfig] = None):
    if torch is None:
        raise RuntimeError("PyTorch is not installed. dasband inference requires torch.")
    base_config = config_override if config_override is not None else DASBandConfig()
    device = resolve_device(base_config.device)
    model, checkpoint_config, _ = load_model_checkpoint(checkpoint_path, device=device)
    model.to(device)

    patch_frames = int(config_override.patch_frames if config_override is not None else checkpoint_config.patch_frames)
    patch_stride = int(config_override.patch_stride if config_override is not None else checkpoint_config.patch_stride)
    indices = build_patch_indices(feature_cube.shape[1], patch_frames, patch_stride)

    acc = np.zeros((feature_cube.shape[1], feature_cube.shape[2]), dtype=np.float32)
    weight = np.zeros_like(acc)
    with torch.no_grad():
        for patch in indices:
            x = torch.from_numpy(feature_cube[:, patch.start : patch.end, :][None, ...]).to(device)
            logits = model(x)
            prob = sigmoid(logits[0, 0].detach().cpu().numpy()).astype(np.float32)
            acc[patch.start : patch.end] += prob
            weight[patch.start : patch.end] += 1.0
    mask = acc / np.maximum(weight, 1e-6)
    return mask.astype(np.float32), checkpoint_config


def run_inference(das_csv: str, checkpoint_path: str, config: DASBandConfig, output_dir: Optional[str] = None):
    out_dir = Path(output_dir) if output_dir else config.resolve_run_root("infer")
    ensure_dir(out_dir)

    das, _ = load_das_csv(das_csv)
    das_trimmed, used_start_s, used_end_s = trim_das(das, config.das_fs, config.trim_start_s, config.trim_end_s)
    feature_cube, feature_names, frame_times, primary_energy = build_feature_cube(das_trimmed, config)
    mask, checkpoint_config = infer_mask(feature_cube, checkpoint_path, config_override=config)

    centroid = weighted_centroid(mask, threshold=config.centroid_threshold)
    measurement_confidence = estimate_measurement_confidence(mask)
    dp_path = extract_path_dp(mask, config)
    kalman_path, kalman_velocity = kalman_smooth_track(centroid, frame_times, measurement_confidence, config)

    if str(config.decode_mode).lower() == "dp":
        path = dp_path
    else:
        path = kalman_path

    path = np.clip(path, 0.0, float(mask.shape[1] - 1)).astype(np.float32)
    sigma = estimate_uncertainty(mask, path, config=config)

    np.save(out_dir / "pred_mask.npy", mask)
    pd.DataFrame(
        {
            "time": frame_times,
            "centroid_channel": centroid,
            "dp_path_channel": dp_path,
            "kalman_path_channel": kalman_path,
            "kalman_velocity": kalman_velocity,
            "path_channel": path,
            "measurement_confidence": measurement_confidence,
            "sigma": sigma,
        }
    ).to_csv(out_dir / "track.csv", index=False)
    save_json(
        out_dir / "infer_summary.json",
        {
            "checkpoint_path": str(checkpoint_path),
            "used_trim_start_s": used_start_s,
            "used_trim_end_s": used_end_s,
            "feature_names": feature_names,
            "checkpoint_config": checkpoint_config.to_dict(),
            "decode_mode": config.decode_mode,
        },
    )
    plot_inference_result(
        primary_energy,
        frame_times,
        mask,
        path,
        sigma,
        str(out_dir / "inference_result.png"),
        centroid=centroid,
        dp_path=dp_path,
    )
    return {
        "output_dir": out_dir,
        "mask": mask,
        "frame_times": frame_times,
        "primary_energy": primary_energy,
        "centroid": centroid,
        "dp_path": dp_path,
        "kalman_path": kalman_path,
        "path": path,
        "sigma": sigma,
    }
