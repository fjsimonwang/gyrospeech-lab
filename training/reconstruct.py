"""用训练好的 U-Net 把新采集的加速度 CSV 重建成语音 WAV（Griffin-Lim 声码器）。"""
import argparse
import numpy as np
import librosa
import soundfile as sf
import torch

from prepare_pairs import load_accel_csv, mel, N_MELS_ACCEL, N_MELS_SPEECH, HOP_S, FRAME_S, SPEECH_SR
from train_reconstruction import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--accel", required=True)
    ap.add_argument("--out", default="recon.wav")
    a = ap.parse_args()

    dev = "cpu"
    net = UNet().to(dev)
    net.load_state_dict(torch.load(a.model, map_location=dev)["state"])
    net.eval()

    sig, rate = load_accel_csv(a.accel)
    accel_mel = mel(sig, rate, N_MELS_ACCEL)                  # [32, T]
    X = torch.from_numpy(accel_mel).unsqueeze(0).unsqueeze(0)  # [1,1,32,T]
    with torch.no_grad():
        pred_db = net(X).squeeze().numpy()                    # [80, T]

    # 反梅尔 + Griffin-Lim
    S = librosa.db_to_power(pred_db)
    hop = int(HOP_S * SPEECH_SR)
    win = int(FRAME_S * SPEECH_SR)
    n_fft = 1 << (win - 1).bit_length()
    wav = librosa.feature.inverse.mel_to_audio(
        S, sr=SPEECH_SR, n_fft=n_fft, hop_length=hop, win_length=win, n_iter=64)
    wav = wav / (np.abs(wav).max() + 1e-9) * 0.9
    sf.write(a.out, wav.astype(np.float32), SPEECH_SR)
    print(f"wrote {a.out}  ({len(wav)/SPEECH_SR:.1f}s)")


if __name__ == "__main__":
    main()
