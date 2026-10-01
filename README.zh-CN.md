# GyroSpeech Lab (Android)

运动传感器声学旁路（acoustic side-channel）的**研究/教育演示** App —
计划文档 [`GYRO_SPEECH_APP_PLAN.md`](GYRO_SPEECH_APP_PLAN.md) 的 **M0** 里程碑实现。

## 功能
- 采集**陀螺仪**或**加速度计**三轴数据（`SENSOR_DELAY_FASTEST` 请求最高速率）。
- 实时显示：真实到达采样率 (Hz)、奈奎斯特上限、时域幅度波形、FFT 幅度谱。
- 一键把带时间戳的原始三轴录制成 **CSV**（存到 App 专属外部目录，供离线训练）。
- 内建合规提示：仅限本人设备 / 经同意的实验。

## 编译运行

需要 Android SDK。两种方式：

**A. Android Studio（推荐）**
1. `File → Open` 选择本目录。
2. 等待 Gradle sync，连真机（不要用模拟器——模拟器没有真实传感器噪声）。
3. Run ▶。

**B. 命令行**
```bash
# 先创建 local.properties 指向你的 SDK（或设置 ANDROID_HOME）
echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties
./gradlew assembleDebug
./gradlew installDebug      # 手机已连接并开启 USB 调试
```
产物：`app/build/outputs/apk/debug/app-debug.apk`

## 使用
把手机平放桌面 → 对着桌面说话或让手机外放播放音频 → 观察波形/频谱随声音变化 → 点"录制"保存 CSV。

导出 CSV：
```bash
adb pull /sdcard/Android/data/com.research.gyrospeech/files/
```

## 现实边界
- Android 12+ 传感器默认封顶 **200 Hz** → 只能捕获 **<100 Hz** 分量。
- 陀螺仪路线：只能做性别/说话人/极少数关键词，**无法逐字转写**。
- 连续语音重建需走**加速度计 + 手机自身外放扬声器**场景（见计划 M4）。

## 版本
- Kotlin 2.0.21 · AGP 8.7.2 · Gradle 8.11.1 · compileSdk 35 · minSdk 26 · Jetpack Compose
