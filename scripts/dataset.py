# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - import guard
    torch = None
    Dataset = object


@dataclass
class PatchIndex:
    start: int
    end: int


def build_patch_indices(total_frames: int, patch_frames: int, stride: int) -> List[PatchIndex]:
    patch_frames = min(int(patch_frames), int(total_frames))
    stride = max(1, int(stride))
    if patch_frames <= 0:
        raise ValueError("patch_frames must be positive.")
    indices = []
    for start in range(0, max(1, total_frames - patch_frames + 1), stride):
        end = min(total_frames, start + patch_frames)
        if end - start < patch_frames:
            start = max(0, total_frames - patch_frames)
            end = total_frames
        idx = PatchIndex(start=start, end=end)
        if not indices or (indices[-1].start != idx.start or indices[-1].end != idx.end):
            indices.append(idx)
    if not indices:
        indices.append(PatchIndex(0, total_frames))
    elif indices[-1].end < total_frames:
        indices.append(PatchIndex(max(0, total_frames - patch_frames), total_frames))
    return indices


def build_center_targets(points_df, frame_times: np.ndarray):
    center = np.full(len(frame_times), np.nan, dtype=np.float32)
    weight = np.zeros(len(frame_times), dtype=np.float32)
    if points_df is None or len(points_df) == 0:
        return center, weight
    for _, row in points_df.iterrows():
        idx = int(np.argmin(np.abs(frame_times - float(row["time"]))))
        if weight[idx] <= 0:
            center[idx] = float(row["channel"])
            weight[idx] = float(row.get("confidence", 1.0))
        else:
            center[idx] = (center[idx] * weight[idx] + float(row["channel"])) / (weight[idx] + 1.0)
            weight[idx] += 1.0
    weight = np.clip(weight, 0.0, 1.0)
    return center, weight


class TimePatchDataset(Dataset):
    def __init__(self, feature_cube, pseudo_label, frame_times, points_df, patch_frames, stride):
        if torch is None:
            raise RuntimeError("PyTorch is required to build the training dataset.")
        self.feature_cube = np.asarray(feature_cube, dtype=np.float32)
        self.pseudo_label = np.asarray(pseudo_label, dtype=np.float32)
        self.frame_times = np.asarray(frame_times, dtype=np.float32)
        self.center_target, self.center_weight = build_center_targets(points_df, self.frame_times)
        self.indices = build_patch_indices(self.feature_cube.shape[1], patch_frames, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        patch = self.indices[idx]
        sl = slice(patch.start, patch.end)
        x = torch.from_numpy(self.feature_cube[:, sl, :])
        y = torch.from_numpy(self.pseudo_label[sl, :][None, :, :])
        center = torch.from_numpy(self.center_target[sl])
        weight = torch.from_numpy(self.center_weight[sl])
        return {
            "x": x,
            "y": y,
            "center_target": center,
            "center_weight": weight,
            "start": patch.start,
            "end": patch.end,
        }
