"""复现 App 内确定性探测音（与 AudioLab.playProbe 一致），作为训练参考音频。"""
import argparse
import numpy as np
import soundfile as sf


def synth(duration_s: float, rate: int = 16000) -> np.ndarray:
    t = np.arange(int(duration_s * rate)) / rate
    carriers = [180.0, 350.0, 620.0]
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)      # 4Hz 节奏
    gate = ((t % 0.8) < 0.6).astype(np.float32)         # 停顿
    s = sum(np.sin(2 * np.pi * fc * t) for fc in carriers) / len(carriers)
    s = s * env * gate
    return (0.9 * s).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--out", default="ref.wav")
    a = ap.parse_args()
    sf.write(a.out, synth(a.duration, a.rate), a.rate)
    print(f"wrote {a.out}  ({a.duration}s @ {a.rate}Hz)")
