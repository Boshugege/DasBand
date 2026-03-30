# DASBand

`dasband/` 是一套围绕“DAS 时空响应带”展开的弱监督建模实现。它不再把问题视为“先做脚步点二分类，再给每个时间点找一个峰值通道”，而是把问题改写为：

1. 先构造一条连续的时空带伪标签。
2. 训练一个 2D mask 网络直接预测 `M(t, c)`。
3. 再从 `M(t, c)` 中解码主轨迹和不确定度。

换句话说，`dasband` 的目标不是找离散的脚步点，而是学习“人沿光纤行走时在 DAS 时空图上形成的连续主响应条带”。

## 一句话理解整体逻辑

主入口 `main.py` 的执行顺序是：

1. 从 `Airtag` 读取目标人的时间范围。
2. 从 `TDMS` 中裁剪出该时间段对应的 DAS CSV。
3. 生成候选点 `(t_k, c_k)`。
4. 对候选点做轨迹级清洗。
5. 把清洗后的点插值成连续中心线。
6. 生成带状伪标签 `T(t, c)`。
7. 训练 2D 分割网络预测 `M(t, c)`。
8. 用质心和动态规划从 `M(t, c)` 提取主轨迹 `c*(t)`。
9. 计算 `sigma(t)` 作为轨迹宽度和不确定度。

## 代码组织

源码结构采用“根目录只保留总入口和说明，所有实现放到 `scripts/`”的方式：

- `main.py`
  作用：总 workflow，按名字串起 TDMS 提取、伪标签准备、训练和推理。
- `scripts/config.py`
  作用：集中定义 `DASBandConfig`，统一管理路径、滤波、清洗、训练和解码参数。
- `scripts/extract_tdms.py`
  作用：按 Airtag 时间窗从 `TDMS` 提取对应 DAS 片段并写成 CSV。
- `scripts/io.py`
  作用：加载 DAS CSV，时间裁剪，多频带滤波，构造特征立方体 `feature_cube`。
- `scripts/candidates.py`
  作用：读取已有候选点 CSV，或者从音频中提取时间点并映射峰值通道。
- `scripts/trajectory_cleaning.py`
  作用：两段鲁棒直线拟合、折返点搜索、离群点剔除、可选投影。
- `scripts/pseudo_label.py`
  作用：插值中心线，生成高斯带/硬带标签，并与信号先验组合。
- `scripts/dataset.py`
  作用：把完整时空图切成时间 patch，供网络训练。
- `scripts/model.py`
  作用：定义轻量 2D U-Net 风格分割网络。
- `scripts/losses.py`
  作用：计算 `mask + center + smooth + tv + area` 组合损失。
- `scripts/pipeline.py`
  作用：串联 `prepare / train / infer` 三个阶段。
- `scripts/decoder.py`
  作用：从预测 mask 中提取质心轨迹、动态规划轨迹和不确定度。
- `scripts/viz.py`
  作用：输出候选点清洗图、伪标签图、推理结果图。
- `scripts/prepare_labels.py`
  作用：单独运行伪标签准备阶段。
- `scripts/train.py`
  作用：单独从准备好的伪标签目录训练模型。
- `scripts/infer.py`
  作用：单独使用训练好的 checkpoint 做推理。

## 主入口的运行顺序

最常用的命令是：

```bash
python -m dasband.main wangdihai
```

这条命令默认执行以下步骤。

### 第 1 步：解析名字和数据路径

输入：

- 目标名字，例如 `wangdihai`
- 默认数据目录：
  - `Data/Airtag`
  - `Data/Audio`
  - `Data/DAS`

输出：

- 解析后的数据路径
- 输出根目录 `output/<name>/`

说明：

- 这一步只做路径和参数准备，不做信号处理。
- 主要调参集中在 `main.py` 的命令行参数里。

### 第 2 步：从 TDMS 提取 DAS CSV

输入：

- `Data/Airtag/<name>.csv`
- `Data/DAS/*.tdms`
- 采样率 `--das_fs`
- Airtag 时间时区偏移 `--csv_utc_offset_hours`
- 跳过前若干通道 `--skip_channels`

方法：

- 先从 Airtag CSV 读取起止时间。
- 再从 TDMS 文件名中解析 UTC 时间。
- 找到与目标时间窗重叠的 TDMS 文件。
- 对每个重叠文件计算样本索引区间。
- 拼接输出成一个连续 CSV。

输出：

- `output/<name>/signals/<name>.csv`

说明：

- 这一阶段由 `scripts/extract_tdms.py` 完成。
- 如果你已经有 CSV，可以用 `--skip_extract` 跳过。

### 第 3 步：构造候选点

输入：

- DAS CSV
- 候选点 CSV，或者音频文件

两种模式：

1. 直接读取已有候选点 CSV
   输入格式要求至少包含：
   - `time`
   - `channel`

2. 音频自动生成候选点
   方法：
   - 对音频带通滤波
   - 提取包络
   - 找脚步候选峰值时间
   - 在 DAS 主能量图上为每个时间点选一个峰值通道

输出：

- 原始候选点表 `candidate_points_raw.csv`

说明：

- 候选点的目标是提供一条粗糙但可清洗的轨迹种子。
- 这一步由 `scripts/candidates.py` 负责。

### 第 4 步：轨迹级清洗

输入：

- 原始候选点 `(t_k, c_k)`

方法：

1. 先按时间排序。
2. 枚举折返点。
3. 在折返点左右两侧分别做鲁棒直线拟合。
4. 用总 Huber 残差选最优折点。
5. 用通道残差阈值剔除离群点。
6. 可选把保留点投影回拟合直线。

输出：

- 清洗后的候选点 `candidate_points_clean.csv`
- 轨迹清洗图 `candidate_cleaning.png`

说明：

- 这一阶段解决的是“候选点稀疏、通道有偏、存在离群点”的问题。
- 这一步对应 `scripts/trajectory_cleaning.py`。

### 第 5 步：伪带标签生成

输入：

- 清洗后的候选点
- DAS 主能量图

方法：

1. 按时间把清洗后的点插值成连续中心线 `centerline(t)`。
2. 沿通道维度在中心线周围生成：
   - 高斯带标签，或
   - 硬阈值带标签
3. 计算信号先验 `P(t, c)`。
4. 最终伪标签为：
   - `T(t, c) = band(t, c) * P(t, c)`，如果启用先验
   - 或 `T(t, c) = band(t, c)`，如果关闭先验

输出：

- `pseudo_label.npy`
- `centerline.npy`
- `prior.npy`
- `pseudo_label.png`

说明：

- 这一阶段是整个弱监督流程的关键，因为它把“离散点监督”转换成了“连续带监督”。
- 对应 `scripts/pseudo_label.py`。

### 第 6 步：特征立方体构造

输入：

- DAS CSV
- 多频带配置
- 时间窗与步长

方法：

- 对 DAS 做多频带滤波。
- 在帧级窗口上构造多个特征图：
  - 对数能量
  - 包络
  - 局部相干性
  - 原始包络
- 最后堆叠成 `feature_cube`，形状为：
  - `[B, T_frames, C]`

输出：

- `feature_cube.npy`
- `frame_times.npy`
- `primary_energy.npy`

说明：

- 这些特征是 2D mask 网络的输入。
- 这一步由 `scripts/io.py` 负责。

### 第 7 步：训练 2D mask 模型

输入：

- `feature_cube.npy`
- `pseudo_label.npy`
- 清洗后的候选点

方法：

1. 把完整时空图按时间切成 patch。
2. 输入轻量 U-Net 风格网络。
3. 输出单通道 mask logits。
4. 损失由以下部分组成：
   - `L_mask`：BCE 分割损失
   - `L_center`：候选点中心一致性
   - `L_smooth`：中心线二阶平滑
   - `L_tv`：mask 的时空 TV
   - `L_area`：抑制全亮

输出：

- `output/<name>/dasband/train/dasband_model.pt`
- `train_history.csv`
- `train_summary.json`

说明：

- 训练阶段默认只依赖 `prepare` 产物，不再重新生成伪标签。
- 这一步对应 `scripts/model.py`、`scripts/losses.py` 和 `scripts/pipeline.py`。

### 第 8 步：推理与轨迹解码

输入：

- DAS CSV
- 训练好的 `dasband_model.pt`

方法：

1. 重新构造 `feature_cube`
2. 分 patch 推理并重建整幅预测 mask
3. 从预测的 `M(t, c)` 中计算：
   - 质心轨迹
   - 动态规划主轨迹
   - `sigma(t)` 不确定度

输出：

- `pred_mask.npy`
- `track.csv`
- `inference_result.png`

其中 `track.csv` 至少包含：

- `time`
- `centroid_channel`
- `path_channel`
- `sigma`

说明：

- 动态规划轨迹通常比单纯质心更稳，更适合后续可视化和导航。
- 这一阶段主要在 `scripts/decoder.py` 中实现。

## 每个阶段的输入输出总结

### A. 提取阶段

输入：

- `Data/Airtag/<name>.csv`
- `Data/DAS/*.tdms`

输出：

- `output/<name>/signals/<name>.csv`

### B. 伪标签准备阶段

输入：

- DAS CSV
- 候选点 CSV 或音频文件

输出：

- `candidate_points_raw.csv`
- `candidate_points_clean.csv`
- `feature_cube.npy`
- `frame_times.npy`
- `primary_energy.npy`
- `prior.npy`
- `pseudo_label.npy`
- `centerline.npy`
- `candidate_cleaning.png`
- `pseudo_label.png`
- `metadata.json`

### C. 训练阶段

输入：

- `prepare` 阶段输出目录

输出：

- `dasband_model.pt`
- `train_history.csv`
- `train_summary.json`

### D. 推理阶段

输入：

- DAS CSV
- `dasband_model.pt`

输出：

- `pred_mask.npy`
- `track.csv`
- `inference_result.png`
- `infer_summary.json`

## 最常用的运行方式

### 1. 一条命令跑完整流程

```bash
python -m dasband.main wangdihai
```

### 2. 带常用调参的完整流程

```bash
python -m dasband.main wangdihai \
  --trim_start_s 40 \
  --trim_end_s 210 \
  --gaussian_sigma_ch 2.5 \
  --clean_outlier_threshold_ch 4 \
  --epochs 50 \
  --learning_rate 5e-4 \
  --model_channels 48
```

### 3. 只跑到伪标签准备

```bash
python -m dasband.main wangdihai --stop_after_prepare
```

### 4. 已有 checkpoint，只做提取和推理

```bash
python -m dasband.main wangjiahui \
  --checkpoint output/wangdihai/dasband/train/dasband_model.pt
```

### 5. 单独调用 prepare

```bash
python -m dasband.scripts.prepare_labels \
  --das_csv output/wangdihai/signals/wangdihai.csv \
  --candidate_csv output/wangdihai/label_peak/wangdihai_label_peak_channels.csv \
  --name wangdihai
```

### 6. 单独调用 train

```bash
python -m dasband.scripts.train \
  --prep_dir output/wangdihai/dasband/prepare \
  --name wangdihai
```

### 7. 单独调用 infer

```bash
python -m dasband.scripts.infer \
  --das_csv output/wangjiahui/signals/wangjiahui.csv \
  --checkpoint output/wangdihai/dasband/train/dasband_model.pt \
  --name wangjiahui
```

## 关键调参建议

- `--trim_start_s --trim_end_s`
  作用：裁掉头尾无效段。通常优先调这个。
- `--skip_channels`
  作用：跳过前若干无效或强噪声通道。
- `--gaussian_sigma_ch`
  作用：控制伪带宽度。偏小更尖锐，偏大更宽松。
- `--clean_outlier_threshold_ch`
  作用：轨迹清洗离群阈值。偏小更严格，偏大更保守。
- `--epochs`
  作用：训练轮数。
- `--learning_rate`
  作用：学习率。
- `--model_channels`
  作用：网络宽度。
- `--dp_jump_penalty --dp_curvature_penalty`
  作用：动态规划路径平滑程度。

## 运行和排查建议

- 第一次跑新数据，建议先执行 `--stop_after_prepare`。
  先看 `candidate_cleaning.png` 和 `pseudo_label.png` 是否合理。
- 如果训练一开始就出现 `nan`，先检查：
  - `prepare` 是否是最新重新生成的
  - `pseudo_label.npy` 是否有限
  - `feature_cube.npy` 是否有限
- 如果轨迹明显偏宽，优先减小 `--gaussian_sigma_ch`。
- 如果清洗后点太少，优先增大 `--clean_outlier_threshold_ch`。
- 如果动态规划轨迹抖动太大，优先增大 `--dp_jump_penalty` 和 `--dp_curvature_penalty`。

## 总结

`dasband` 的核心思想可以概括为：

先把“脚步点”整理成一条可用的弱监督带标签，再让网络学习整幅时空图中的主响应条带，最后从预测条带中解码一条最可信、最平滑的主轨迹和它的时变不确定度。
