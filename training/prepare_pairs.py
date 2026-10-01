"""把加速度 CSV 与参考音频对齐、算梅尔谱、切成训练对 (accel_mel, speech_mel)。"""
import argparse
import glob
import os
import numpy as np
import librosa

SPEECH_SR = 16000
N_MELS_ACCEL = 32
N_MELS_SPEECH = 80
FRAME_S = 0.032   # 32ms 帧，两侧统一时间轴
HOP_S = 0.016
CHUNK_FRAMES = 128  # 每个训练样本的帧数


def load_accel_csv(path):
    """返回 (mag[N], rate_hz)。CSV: timestamp_ns,x,y,z。"""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    ts = data[:, 0]
    xyz = data[:, 1:4]
    mag = np.linalg.norm(xyz, axis=1)
    mag = mag - mag.mean()
    dur = (ts[-1] - ts[0]) / 1e9
    rate = (len(ts) - 1) / dur if dur > 0 else 200.0
    return mag.astype(np.float32), float(rate)


def mel(sig, sr, n_mels):
    hop = max(1, int(HOP_S * sr))
    win = max(hop * 2, int(FRAME_S * sr))
    n_fft = 1 << (win - 1).bit_length()
    S = librosa.feature.melspectrogram(
        y=sig, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
        n_mels=n_mels, fmax=sr / 2)
    return librosa.power_to_db(S + 1e-10).astype(np.float32)  # [n_mels, T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", nargs="+", required=True, help="加速度 CSV（可多个/通配）")
    ap.add_argument("--ref", required=True, help="参考音频 WAV")
    ap.add_argument("--out", default="pairs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ref, _ = librosa.load(a.ref, sr=SPEECH_SR, mono=True)
    speech_mel = mel(ref, SPEECH_SR, N_MELS_SPEECH)  # [80, Ts]

    files = []
    for p in a.accel:
        files.extend(glob.glob(p))
    idx = 0
    for f in files:
        sig, rate = load_accel_csv(f)
        accel_mel = mel(sig, rate, N_MELS_ACCEL)      # [32, Ta]
        # 时间轴插值到与语音相同帧数
        T = min(accel_mel.shape[1], speech_mel.shape[1])
        acc = librosa.util.fix_length(accel_mel, size=T, axis=1)
        spe = librosa.util.fix_length(speech_mel, size=T, axis=1)
        # 切成定长块
        for s in range(0, T - CHUNK_FRAMES + 1, CHUNK_FRAMES // 2):
            X = acc[:, s:s + CHUNK_FRAMES]
            Y = spe[:, s:s + CHUNK_FRAMES]
            np.savez(os.path.join(a.out, f"pair_{idx:05d}.npz"), X=X, Y=Y)
            idx += 1
    print(f"wrote {idx} pairs to {a.out}/  (accel_mel {N_MELS_ACCEL} -> speech_mel {N_MELS_SPEECH})")


if __name__ == "__main__":
    main()
