"""U-Net 回归：加速度梅尔谱(32) -> 语音梅尔谱(80)。"""
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Pairs(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(f"{folder}/*.npz"))
        assert self.files, f"no .npz in {folder}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        X = torch.from_numpy(d["X"]).unsqueeze(0)  # [1,32,T]
        Y = torch.from_numpy(d["Y"]).unsqueeze(0)  # [1,80,T]
        return X, Y


class UNet(nn.Module):
    """沿频率维把 32 频带上采样到 80，时间维保持。"""
    def __init__(self, in_mels=32, out_mels=80):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.enc1 = blk(1, 32)
        self.enc2 = blk(32, 64)
        self.pool = nn.MaxPool2d((2, 1))          # 只在频率维下采样
        self.mid = blk(64, 128)
        self.up = nn.Upsample(scale_factor=(2, 1), mode="nearest")
        self.dec2 = blk(128 + 64, 64)
        self.dec1 = blk(64 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.out_mels = out_mels

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.mid(self.pool(e2))
        d2 = self.dec2(torch.cat([self._match(self.up(m), e2), e2], 1))
        d1 = self.dec1(torch.cat([self._match(self.up(d2), e1), e1], 1))
        y = self.head(d1)
        # 频率维线性插值到 out_mels
        y = torch.nn.functional.interpolate(y, size=(self.out_mels, y.shape[-1]), mode="bilinear", align_corners=False)
        return y

    @staticmethod
    def _match(a, ref):
        if a.shape[-2:] != ref.shape[-2:]:
            a = torch.nn.functional.interpolate(a, size=ref.shape[-2:], mode="nearest")
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="model.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dl = DataLoader(Pairs(a.pairs), batch_size=a.bs, shuffle=True)
    net = UNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.L1Loss()

    for ep in range(a.epochs):
        tot = 0.0
        for X, Y in dl:
            X, Y = X.to(dev), Y.to(dev)
            opt.zero_grad()
            loss = lossf(net(X), Y)
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"epoch {ep+1}/{a.epochs}  L1={tot/len(dl):.4f}")
    torch.save({"state": net.state_dict()}, a.out)
    print(f"saved {a.out}  (device={dev})")


if __name__ == "__main__":
    main()
