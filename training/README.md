# M4 离线重建训练脚手架

把 App 采集的 **加速度信号** 学到 **语音频谱**，重建可听（乃至部分可懂）的语音。
这是 App 内 DSP 基线（韵律/音高级）的升级路径——需要你自己采集成对数据 + 训练。

## 原理
```
加速度 CSV (≤200Hz, 时域)  ──STFT──►  加速度对数梅尔谱 X
                                              │  U-Net 回归
参考音频 WAV (16kHz, 干净)  ──STFT──►  语音对数梅尔谱 Y  ◄──── 监督目标
                                              │
                            预测 Ŷ ──Griffin-Lim/声码器──► 重建波形 ──ASR(可选)──► 文字
```
> 现实边界：加速度有效带宽 <100Hz，只能恢复低频包络/共振峰趋势。
> 逐字可懂需要**大量成对数据 + 长时训练**（参考 AccEar 的条件 GAN + 神经声码器），
> 本脚手架给的是可跑通的回归基线，不保证逐字。

## 采集成对数据
两种方式二选一：
1. **同机外放**：用 App「▶ 探测音+采集」，同时另存被播放的参考音频（探测音是确定性的，可由 `synth_probe.py` 复现）。
2. **真实语音**：手机外放一段已知语音 WAV，用 App「● 仅采集」录加速度；参考 WAV 即监督目标。

导出加速度 CSV：
```bash
adb -s <serial> pull /sdcard/Android/data/com.research.gyrospeech/files/ ./data_raw
```
CSV 格式：`timestamp_ns,x,y,z`（App「录制原始数据到 CSV」产出）。

## 步骤
```bash
pip install -r requirements.txt
# 1. 对齐并切片成 (accel_mel, speech_mel) 训练对
python prepare_pairs.py --accel data_raw/sensor_ACCEL_*.csv --ref data_raw/ref.wav --out pairs/
# 2. 训练 U-Net 回归
python train_reconstruction.py --pairs pairs/ --epochs 100 --out model.pt
# 3. 用模型重建 + Griffin-Lim 还原波形
python reconstruct.py --model model.pt --accel data_raw/sensor_ACCEL_new.csv --out recon.wav
```

## 文件
- `synth_probe.py` — 复现 App 内确定性探测音，作参考音频。
- `prepare_pairs.py` — 重采样对齐 + 梅尔谱 + 切片。
- `train_reconstruction.py` — U-Net 加速度谱→语音谱回归。
- `reconstruct.py` — 推理 + Griffin-Lim 声码器。
- `export_tflite.py` — 导出 TFLite，供端上推理（回填到 App）。
