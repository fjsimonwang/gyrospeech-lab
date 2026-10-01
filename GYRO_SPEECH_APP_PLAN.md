# 手机陀螺仪/运动传感器语音旁路解码 App — 技术计划

> 状态：规划稿 · 日期：2026-08-06
> 定位：**研究/教育/防御性安全**演示工具，用于复现并可视化 MEMS 运动传感器的声学旁路（acoustic side-channel）现象，并评估防御。

---

## 0. 现实边界（先读这一节）

物理天花板由采样率决定：

- Android 12 (API 31)+ 强制运动传感器上限 **200 Hz**（未持 `HIGH_SAMPLING_RATE_SENSORS` 权限时）。
- iOS 公开 API：`CMMotionManager` 陀螺仪/加速度计实际稳定上限约 **100 Hz**（硬件可到 800–1600 Hz，但 API 不暴露）。
- 奈奎斯特：200 Hz 采样 → 可还原频率 < 100 Hz。人声可懂度关键频段（辅音、共振峰 1–4 kHz）**物理上采不到**。

因此本 App 的现实目标不是"通用窃听器"，而是：

1. **可复现**：在受控条件下复现 Gyrophone / Spearphone / AccEar 类实验。
2. **可量化**：给出性别识别、说话人识别、关键词识别的准确率曲线。
3. **可防御**：演示 Android 12 限速前后效果差异，作为隐私教育工具。

能做 vs 做不到：

| 目标 | 传感器 | 声源 | 现实性 |
|---|---|---|---|
| 性别分类 | 陀螺仪/加速度计 | 空气传播 | ✅ ~90% |
| 说话人识别（闭集，N≤10） | 陀螺仪/加速度计 | 空气传播 | ✅ ~80% |
| 孤立数字/唤醒词识别 | 陀螺仪 | 空气传播 | ⚠️ 单人~65%，跨人差 |
| 连续语音波形重建 | **加速度计** | **手机自身外放扬声器** | ⚠️ 部分（GAN） |
| 任意对话逐字转文字 | 任一 | 空气传播 | ❌ 当前不可行 |

---

## 1. 两条技术路线

### 路线 A — 陀螺仪 · 空气传播（Gyrophone 复现）
- 手机平放桌面，旁人说话的声压使桌面/机身微振，MEMS 陀螺仪的科里奥利质量块被扰动。
- 采到的是极窄低频带（<100 Hz）：主要是基频与低频包络。
- 适合：性别、说话人、少量关键词。**不适合**逐字转写。

### 路线 B — 加速度计 · 同机扬声器（Spearphone / AccEar 复现）
- 手机外放（通话/语音消息/媒体）时，扬声器振动通过 PCB/外壳**直接机械耦合**到加速度计。
- 信号信噪比远高于路线 A，深度学习可做到接近可懂的语音重建（AccEar 用条件 GAN 做无约束词汇）。
- 适合：演示"同机媒体音频泄漏"，这是最接近"转文字"的路径。

> 建议 MVP 先做**路线 A 的性别+说话人分类**（工程量小、结论稳），再做**路线 B 的关键词/重建**（更接近"转文字"卖点）。

---

## 2. 信号处理与算法链

```
原始三轴信号 (x,y,z @ ~100–200Hz)
  → 去趋势/去重力 (high-pass ~2–5Hz)
  → 分帧 (窗长 250–500ms, 50% overlap, Hamming)
  → 特征提取
       ├─ STFT / 对数功率谱
       ├─ 低频段 MFCC（少数 mel 滤波器，因带宽极窄）
       ├─ 谱质心、过零率、带能量比
       └─ (可选) 三轴融合 / PCA
  → 模型
       ├─ 分类任务: CNN / CRNN / 轻量 Transformer → 性别/说话人/关键词
       └─ 重建任务(路线B): Encoder–Decoder + 条件GAN → 频谱 → 声码器(HiFi-GAN)还原波形
  → (重建后) 常规 ASR (Whisper 等) → 文字
```

### 2.1 预处理关键点
- **重力/漂移去除**：high-pass 滤波，去掉 DC 和低频漂移。
- **传感器混叠利用**：Gyrophone 论文核心技巧——不同机型 ADC 实际过采样特性不同，某些机型混叠会把 >100Hz 的分量折叠回可见带，可"意外"带回部分高频信息。需按机型标定。
- **多机聚合**：论文用两台并排手机分别采样、相位错开，等效提升有效采样率（>200Hz 等效）。可作为进阶特征。

### 2.2 特征
- 分类任务：对数梅尔谱 / 低频 MFCC + 手工统计特征拼接。
- 重建任务：STFT 幅度谱作为网络输入/输出，相位用 Griffin-Lim 或神经声码器恢复。

### 2.3 模型选型
- **性别/说话人分类**：1D/2D-CNN 或 CRNN；数据少时用轻量模型 + 强数据增强。
- **关键词识别**：CRNN + CTC，或小词表分类头。
- **语音重建（路线B）**：参照 AccEar——U-Net/条件 GAN 从加速度谱映射到语音谱，接 HiFi-GAN 声码器，末端接开源 ASR（Whisper small）转文字。

---

## 3. App 架构

### 3.1 采集层（原生，必须原生才能拿到高采样率）
- **iOS**：`CMMotionManager`，`gyroUpdateInterval = 0.01`（100Hz），`startGyroUpdates`/`startAccelerometerUpdates`。后台采集受限，需前台或特定后台模式。
- **Android**：`SensorManager.registerListener` + `SENSOR_DELAY_FASTEST`；Android 12+ 默认封顶 200Hz；如需更高需 `HIGH_SAMPLING_RATE_SENSORS` 权限（且系统仍可能拒绝）。
- 统一输出：带时间戳的三轴序列，写入环形缓冲区。

### 3.2 处理层
- 端上轻量推理：Core ML (iOS) / TFLite (Android) 跑分类模型，实时出性别/说话人标签。
- 重建/转写这类重模型：录段后上传到本地/私有服务器批处理（Python: numpy/scipy/librosa + PyTorch）。

### 3.3 前端/可视化
- 实时三轴波形 + 频谱瀑布图。
- 分类结果面板（性别/说话人置信度）。
- "防御演示"开关：对比限速前后频谱可分性。
- 明确的合规横幅与录音同意流程。

### 3.4 训练与数据管线（离线）
- Python 服务：数据入库 → 特征 → 训练 → 导出 Core ML/TFLite。
- 数据集：自采（受同意的志愿者）+ 公开语音库（如数字/命令词）在受控扬声器下重放采集，做配对训练。

---

## 4. 里程碑

| 阶段 | 内容 | 产出 | 预估 |
|---|---|---|---|
| M0 | 采集 SDK（iOS+Android），验证真实采样率 | 原始信号可视化 App | 1–2 周 |
| M1 | 数据采集协议 + 标注管线（受同意志愿者） | 配对数据集 v1 | 1–2 周 |
| M2 | 路线A：性别/说话人分类基线 | 准确率报告 + 端上模型 | 2–3 周 |
| M3 | 路线A：孤立关键词识别 | CRNN+CTC 结果 | 2–3 周 |
| M4 | 路线B：同机扬声器语音重建 + ASR 转文字 | 重建 demo + WER 报告 | 4–6 周 |
| M5 | 防御演示 + 合规打包 | 教育版发布 | 2 周 |

---

## 5. 法律与伦理合规（硬约束）

- **本工具用途限定为**：本人设备上的研究、经明确同意的实验、课堂/安全演示、防御评估。
- 未经同意采集他人语音在多数法域构成非法窃听/侵犯隐私（如美国 Wiretap Act、中国《个人信息保护法》/《民法典》隐私权）。
- App 内必须有：显式录音同意、数据本地化选项、随时删除、清晰说明"这是隐私风险演示工具"。
- **不做**：隐蔽后台窃听、绕过权限、去标识化规避、面向第三方的定向监听。
- 上架风险：App Store/Play 对"监听/间谍"类应用有严格审查，需以"传感器隐私研究/教育"定位并提供来源论文。

---

## 6. 主要风险

- **技术风险**：路线A可懂度天花板低；跨机型/跨说话人泛化差；Android 12 限速削弱效果。
- **合规风险**：极易被误用，需在设计层面加约束（同意流程、仅本机、日志透明）。
- **上架风险**：可能被应用商店拒。建议先做内部研究工具/开源复现，不急于商用上架。

---

## 7. 参考研究

- Gyrophone: Recognizing Speech from Gyroscope Signals (USENIX Security 2014) — Michalevsky, Boneh, Nakibly
- Spearphone: Accelerometer-Sensed Reverberations from Smartphone Loudspeakers (WiSec 2021)
- AccEar: Accelerometer Acoustic Eavesdropping with Unconstrained Vocabulary (2022, arXiv 2212.01042)
- Learning-based Practical Smartphone Eavesdropping with Built-in Accelerometer (NDSS 2020)
- EarSpy: Spying Caller Speech via Ear Speaker Vibrations (2022, arXiv 2212.12151)
- Android 12 传感器 200Hz 限速：developer.android.com sensors_overview
