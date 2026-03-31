# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from .config import DASBandConfig


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def weighted_centroid(mask: np.ndarray, threshold: float = 0.0):
    mask = np.asarray(mask, dtype=np.float64)
    if threshold > 0:
        mask = np.where(mask >= threshold, mask, 0.0)
    channels = np.arange(mask.shape[1], dtype=np.float64)[None, :]
    numer = np.sum(mask * channels, axis=1)
    denom = np.sum(mask, axis=1) + 1e-8
    return (numer / denom).astype(np.float32)


def estimate_measurement_confidence(mask: np.ndarray):
    mask = np.asarray(mask, dtype=np.float64)
    peak = np.max(mask, axis=1)
    mass = np.sum(mask, axis=1)
    concentration = peak / (mass + 1e-6)
    conf = 0.7 * peak + 0.3 * np.clip(20.0 * concentration, 0.0, 1.0)
    return np.clip(conf, 0.05, 0.95).astype(np.float32)


def extract_path_dp(mask: np.ndarray, config: DASBandConfig):
    prob = np.clip(np.asarray(mask, dtype=np.float64), 1e-6, 1.0)
    emission = np.log(prob)
    T, C = emission.shape
    max_jump = max(1, int(config.dp_max_jump_ch))
    lam = float(config.dp_jump_penalty)

    score = np.full((T, C), -np.inf, dtype=np.float64)
    prev = np.full((T, C), -1, dtype=np.int32)
    score[0] = emission[0]

    for t in range(1, T):
        for c in range(C):
            lo = max(0, c - max_jump)
            hi = min(C, c + max_jump + 1)
            prev_cands = np.arange(lo, hi)
            candidate_score = score[t - 1, prev_cands] - lam * np.abs(prev_cands - c)
            if t >= 2 and config.dp_curvature_penalty > 0:
                ref = prev[t - 1, prev_cands]
                valid_ref = ref >= 0
                curvature = np.zeros_like(candidate_score)
                curvature[valid_ref] = float(config.dp_curvature_penalty) * np.abs(c - 2 * prev_cands[valid_ref] + ref[valid_ref])
                candidate_score = candidate_score - curvature
            best_idx = int(np.argmax(candidate_score))
            score[t, c] = emission[t, c] + candidate_score[best_idx]
            prev[t, c] = int(prev_cands[best_idx])

    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = max(0, int(prev[t, path[t]]))
    return path.astype(np.float32)


def kalman_smooth_track(measurements: np.ndarray, frame_times: np.ndarray, measurement_confidence: np.ndarray, config: DASBandConfig):
    z = np.asarray(measurements, dtype=np.float64)
    t = np.asarray(frame_times, dtype=np.float64)
    conf = np.clip(np.asarray(measurement_confidence, dtype=np.float64), 0.05, 0.95)
    n = len(z)
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    x_filt = np.zeros((n, 2), dtype=np.float64)
    p_filt = np.zeros((n, 2, 2), dtype=np.float64)
    x_pred = np.zeros((n, 2), dtype=np.float64)
    p_pred = np.zeros((n, 2, 2), dtype=np.float64)

    x_filt[0] = np.array([z[0], 0.0], dtype=np.float64)
    p_filt[0] = np.diag([float(config.kalman_init_pos_var), float(config.kalman_init_vel_var)])

    for i in range(1, n):
        dt = max(1e-6, float(t[i] - t[i - 1]))
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = float(config.kalman_process_var)
        Q = q * np.array(
            [[0.25 * dt ** 4, 0.5 * dt ** 3], [0.5 * dt ** 3, dt ** 2]],
            dtype=np.float64,
        )

        x_pred[i] = F @ x_filt[i - 1]
        p_pred[i] = F @ p_filt[i - 1] @ F.T + Q

        H = np.array([[1.0, 0.0]], dtype=np.float64)
        R = float(config.kalman_measurement_var_floor) + float(config.kalman_measurement_var) / (conf[i] ** 2)
        S = H @ p_pred[i] @ H.T + np.array([[R]], dtype=np.float64)
        K = p_pred[i] @ H.T @ np.linalg.inv(S)
        innovation = np.array([z[i] - (H @ x_pred[i])[0]], dtype=np.float64)
        x_filt[i] = x_pred[i] + (K @ innovation).reshape(-1)
        p_filt[i] = (np.eye(2, dtype=np.float64) - K @ H) @ p_pred[i]

    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()
    for i in range(n - 2, -1, -1):
        dt = max(1e-6, float(t[i + 1] - t[i]))
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        c = p_filt[i] @ F.T @ np.linalg.inv(p_pred[i + 1] + 1e-9 * np.eye(2))
        x_smooth[i] = x_filt[i] + c @ (x_smooth[i + 1] - x_pred[i + 1])
        p_smooth[i] = p_filt[i] + c @ (p_smooth[i + 1] - p_pred[i + 1]) @ c.T

    return x_smooth[:, 0].astype(np.float32), x_smooth[:, 1].astype(np.float32)


def estimate_uncertainty(mask: np.ndarray, path: np.ndarray, config: DASBandConfig | None = None):
    prob = np.asarray(mask, dtype=np.float64)
    channels = np.arange(prob.shape[1], dtype=np.float64)[None, :]
    path = np.asarray(path, dtype=np.float64)[:, None]
    numer = np.sum(((channels - path) ** 2) * prob, axis=1)
    denom = np.sum(prob, axis=1) + 1e-8
    sigma = np.sqrt(numer / denom).astype(np.float32)
    if config is not None:
        sigma = sigma * float(config.sigma_scale)
        sigma = np.clip(sigma, float(config.sigma_min), float(config.sigma_max))
    return sigma.astype(np.float32)


class TrackState:
    INIT = "INIT"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    DEAD = "DEAD"


class Track:
    def __init__(self, track_id: int, start_time: float, start_frame: int, start_pos: float, config: DASBandConfig):
        self.track_id = track_id
        self.state = TrackState.INIT
        
        self.hits = 1
        self.time_since_update = 0
        
        # 卡尔曼状态：[位置(channel), 速度(channel/s)]
        self.x = np.array([start_pos, 0.0], dtype=np.float64)
        self.P = np.diag([float(config.kalman_init_pos_var), float(config.kalman_init_vel_var)])
        self.q = float(config.kalman_process_var)
        self.r = float(config.kalman_measurement_var)
        self.r_floor = float(config.kalman_measurement_var_floor)
        self.damping = getattr(config, 'mot_damping', 0.9)
        
        # 记录生命周期历史
        self.history = [{
            "frame": start_frame,
            "time": start_time,
            "channel": start_pos,
            "state": self.state
        }]
        
    def predict(self, dt: float):
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        Q = self.q * np.array([
            [0.25 * dt**4, 0.5 * dt**3],
            [0.5 * dt**3, dt**2]
        ], dtype=np.float64)
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        
    def apply_damping(self):
        # 物理学阻尼：失去观测时速度按比例缓释，预测的不确定性膨胀 (暂离态物理惯性)
        self.x[1] *= self.damping
        self.P[0, 0] += self.q * 2.0
        
    def update(self, measurement: float):
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        R = np.array([[self.r + self.r_floor]], dtype=np.float64)
        
        y = measurement - (H @ self.x)[0]
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + (K @ np.array([y])).reshape(-1)
        self.P = (np.eye(2) - K @ H) @ self.P

    def record(self, frame_idx: int, t_val: float):
        self.history.append({
            "frame": frame_idx,
            "time": t_val,
            "channel": float(self.x[0]),
            "state": self.state
        })


def extract_mot_tracks(mask: np.ndarray, frame_times: np.ndarray, config: DASBandConfig):
    """
    MOT (Multi-Object Tracking) 入口函数
    基于找峰(Peak Detection) + 匈牙利算法数据关联 + 四元状态卡尔曼生存周期的对象追踪器
    """
    T, C = mask.shape
    threshold = getattr(config, 'mot_peak_threshold', 0.2)
    init_hits = getattr(config, 'mot_init_hits', 3)
    max_age = getattr(config, 'mot_max_age', 15)
    
    # 【修复1：物理限制卡口 (考虑到触发脚印的声学空间离散性)】
    # 一帧时间虽短，人绝对走不了这么远。但这不仅是“人”的位移，更是“声源”的位移。
    # 重踩下去，应力波可能在前方甚至后方几米处诱发另一个子频带的高光，形成“虚拟脚印偏移”。
    # 所以模型看到的波峰，并不完全等于人身体质心的物理位移，而是“这个区域内的脚步激发现象”。
    # 结合你“触发识别可能在附近五个通道”的先验：
    match_thresh = 5.0 # max dx allowed per frame, 容忍由于步伐引起的相邻通道信号横跳
    
    # 增加最小峰距离，同一个人的一步不可能在相距仅仅 1-3 米（通道）同时产生两个并列的独立人
    # 如果允许单步脚印在附件 5 个通道内散布，那么必须用 5.0 把这些散布的“碎步峰”统一 NMS 掉，
    # 否则同一次踩踏引发的两个波峰会被建出两个 Track。
    min_peak_dist = 5.0
    
    tracks = []
    active_tracks = []
    next_track_id = 1
    
    for t_idx in range(T):
        t_val = frame_times[t_idx]
        dt = float(t_val - frame_times[t_idx-1]) if t_idx > 0 else 0.025
        
        # 1. 独立并发波峰提取 (Detection)带上了强非极大值抑制 (NMS)
        m_t = mask[t_idx]
        peaks, props = find_peaks(m_t, height=threshold, distance=max(1, int(min_peak_dist)))
        
        # 【修复2：亚像素提取与重力中心化合成 (People Synthesis)】
        # 纯粹的波峰只是一个整数通道。很多时候一个人踩下去，周围几个通道都有响应。
        # 这里使用与你设定“5通道物理激发范围”相匹配的合成半径。
        # win_radius = 2 代表提取峰值及左右各2个通道，总计正好是 5 个通道 (即5米)。
        # 这保证了这5米内所有的概率亮斑被绝对公平地加权平均缩为一个极具代表性的实数坐标。
        detections = []
        for p in peaks:
            # 在峰值附近提取小窗口求质心
            win_radius = 2  # 局部合成半径 [-2, -1, 0, 1, 2] 共 5 个通道
            start_idx = max(0, p - win_radius)
            end_idx = min(C, p + win_radius + 1)
            local_mask = m_t[start_idx:end_idx]
            local_channels = np.arange(start_idx, end_idx, dtype=np.float64)
            if np.sum(local_mask) > 1e-8:
                centroid = np.sum(local_mask * local_channels) / np.sum(local_mask)
                detections.append(centroid)
            else:
                detections.append(float(p))
                
        detections = np.array(detections, dtype=np.float64)
        
        # 2. 预测存活状态轨 (Prediction)
        for trk in active_tracks:
            trk.predict(dt)
            
        matched_tracks = []
        matched_detections = []
        
        # 3. 匈牙利最优匹配 (Data Association)
        if len(active_tracks) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(active_tracks), len(detections)))
            for i, trk in enumerate(active_tracks):
                for j, det in enumerate(detections):
                    cost_matrix[i, j] = np.abs(trk.x[0] - det)
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= match_thresh:
                    matched_tracks.append(r)
                    matched_detections.append(c)
                    
        # 4. 更新成功匹配的 Track
        for match_i, trk_idx in enumerate(matched_tracks):
            trk = active_tracks[trk_idx]
            det_idx = matched_detections[match_i]
            trk.update(detections[det_idx])
            trk.hits += 1
            trk.time_since_update = 0
            
            if trk.state == TrackState.INIT and trk.hits >= init_hits:
                trk.state = TrackState.CONFIRMED
            elif trk.state == TrackState.LOST:
                trk.state = TrackState.CONFIRMED
                
        # 5. 处理丢失的 Track (LOST / DEAD / Coasting)
        unmatched_tracks = [i for i in range(len(active_tracks)) if i not in matched_tracks]
        for idx in unmatched_tracks:
            trk = active_tracks[idx]
            trk.time_since_update += 1
            trk.apply_damping()
            
            if trk.state == TrackState.INIT:
                if trk.time_since_update > 1: # INIT态未能连续验证，直接枪毙
                    trk.state = TrackState.DEAD
            elif trk.state == TrackState.CONFIRMED:
                trk.state = TrackState.LOST # 丢失目标，转入寻找重连期
            
            if trk.state == TrackState.LOST and trk.time_since_update > max_age:
                trk.state = TrackState.DEAD # 长时间未重连，彻底宣告消失
                
        # 6. 为未匹配的游离点建立新 Track
        unmatched_detections = [j for j in range(len(detections)) if j not in matched_detections]
        for j in unmatched_detections:
            new_trk = Track(next_track_id, float(t_val), t_idx, detections[j], config)
            tracks.append(new_trk)
            active_tracks.append(new_trk)
            next_track_id += 1
            
        # 7. 清理 DEAD 目标
        active_tracks = [t for t in active_tracks if t.state != TrackState.DEAD]
            
        # 8. 记录这一帧活跃轨迹的快照
        for trk in active_tracks:
            # 记录历史轨迹，第一帧初始化时已写记录，防止重复
            if t_idx > trk.history[0]["frame"]:
                trk.record(t_idx, float(t_val))
                
    # ================= 组织输出为长列 DataFrame =================
    all_records = []
    for trk in tracks:
        # 只保留至少被"确认"过的主轴轨迹
        is_valid = any(r["state"] == TrackState.CONFIRMED for r in trk.history)
        if is_valid:
            for r in trk.history:
                all_records.append({
                    "track_id": trk.track_id,
                    "frame": r["frame"],
                    "time": r["time"],
                    "channel": r["channel"],
                    "state": r["state"]
                })
                
    if len(all_records) > 0:
        df_tracks = pd.DataFrame(all_records)
    else:
        df_tracks = pd.DataFrame(columns=["track_id", "frame", "time", "channel", "state"])
        
    return df_tracks
