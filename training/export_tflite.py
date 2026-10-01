"""导出模型供端上推理。

PyTorch → TFLite 没有稳定的直连路径，这里走可靠的 ONNX 导出；
再用 onnx2tf / onnx-tensorflow 转 TFLite 回填到 App。
"""
import argparse
import torch
from train_reconstruction import UNet

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--frames", type=int, default=128)
    a = ap.parse_args()

    net = UNet()
    net.load_state_dict(torch.load(a.model, map_location="cpu")["state"])
    net.eval()
    dummy = torch.randn(1, 1, 32, a.frames)  # [B,1,accel_mels,T]
    torch.onnx.export(
        net, dummy, a.out,
        input_names=["accel_mel"], output_names=["speech_mel"],
        dynamic_axes={"accel_mel": {3: "time"}, "speech_mel": {3: "time"}},
        opset_version=17)
    print(f"wrote {a.out}")
    print("再转 TFLite：  pip install onnx2tf && onnx2tf -i", a.out, "-o tflite_out/")
