# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_candidate_cleaning(primary_energy, frame_times, raw_points_df: pd.DataFrame, clean_points_df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(
        np.log1p(primary_energy.T),
        aspect="auto",
        origin="lower",
        extent=[frame_times[0], frame_times[-1], 0, primary_energy.shape[1] - 1],
        cmap="viridis",
    )
    if len(raw_points_df) > 0:
        ax.scatter(raw_points_df["time"], raw_points_df["channel"], s=30, c="white", alpha=0.7, label="raw")
    if len(clean_points_df) > 0:
        ax.scatter(clean_points_df["time"], clean_points_df["channel"], s=20, c="red", alpha=0.9, label="clean")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title("Candidate Cleaning")
    ax.legend(loc="upper right")
    _savefig(Path(output_path))


def plot_pseudo_label(primary_energy, frame_times, pseudo_label, centerline, output_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axes[0].imshow(
        np.log1p(primary_energy.T),
        aspect="auto",
        origin="lower",
        extent=[frame_times[0], frame_times[-1], 0, primary_energy.shape[1] - 1],
        cmap="viridis",
    )
    axes[0].plot(frame_times, centerline, color="white", linewidth=1.2)
    axes[0].set_ylabel("Channel")
    axes[0].set_title("Primary Energy + Centerline")
    axes[1].imshow(
        pseudo_label.T,
        aspect="auto",
        origin="lower",
        extent=[frame_times[0], frame_times[-1], 0, pseudo_label.shape[1] - 1],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Channel")
    axes[1].set_title("Pseudo Band Label")
    _savefig(Path(output_path))


def plot_inference_result(primary_energy, frame_times, mask, path, sigma, output_path: str, centroid=None, dp_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axes[0].imshow(
        np.log1p(primary_energy.T),
        aspect="auto",
        origin="lower",
        extent=[frame_times[0], frame_times[-1], 0, primary_energy.shape[1] - 1],
        cmap="viridis",
    )
    axes[0].plot(frame_times, path, color="white", linewidth=1.5)
    if centroid is not None:
        axes[0].plot(frame_times, centroid, color="gold", linewidth=1.0, alpha=0.8, linestyle="--")
    if dp_path is not None:
        axes[0].plot(frame_times, dp_path, color="cyan", linewidth=0.8, alpha=0.6)
    axes[0].fill_between(frame_times, path - sigma, path + sigma, color="white", alpha=0.2)
    axes[0].set_ylabel("Channel")
    axes[0].set_title("Primary Energy + Decoded Track")
    axes[1].imshow(
        mask.T,
        aspect="auto",
        origin="lower",
        extent=[frame_times[0], frame_times[-1], 0, mask.shape[1] - 1],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].plot(frame_times, path, color="cyan", linewidth=1.2)
    if centroid is not None:
        axes[1].plot(frame_times, centroid, color="white", linewidth=0.9, alpha=0.7, linestyle="--")
    if dp_path is not None:
        axes[1].plot(frame_times, dp_path, color="lime", linewidth=0.8, alpha=0.5)
    axes[1].fill_between(frame_times, path - sigma, path + sigma, color="cyan", alpha=0.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Channel")
    axes[1].set_title("Predicted Mask")
    _savefig(Path(output_path))
