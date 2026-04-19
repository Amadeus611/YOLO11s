"""Latency / FPS benchmark for detection models.

Measures:
  * parameters        (M)
  * FLOPs             (G)   -- via ultralytics.utils.torch_utils.get_flops
  * per-image latency (ms)  -- averaged over ``--iters`` forward passes
  * throughput FPS          -- 1000 / latency

The benchmark only times ``model(x)`` (pure PyTorch forward). It does NOT
include image I/O, preprocessing, NMS, or postprocessing. For an end-to-end
pipeline benchmark, use ``yolo benchmark model=... format=torch data=...``.

Usage
-----
GPU benchmark at imgsz=640, batch=1::

    python scripts/bench_speed.py yolo11s-pvrp.yaml --device 0 --imgsz 640

CPU benchmark at batch=1 (no CUDA required)::

    python scripts/bench_speed.py yolo11s-pvrp-lite.yaml --device cpu

Benchmark an exported ONNX via ORT is intentionally out of scope; export
and benchmark separately with ``yolo benchmark``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, select_device  # noqa: E402


def benchmark(
    model_path: str,
    imgsz: int = 640,
    batch: int = 1,
    device: str = "0",
    warmup: int = 20,
    iters: int = 200,
    half: bool = False,
) -> dict:
    dev = select_device(device, verbose=False)
    m = YOLO(model_path, task="detect")
    net = m.model.to(dev).eval()

    # Count parameters and FLOPs in fp32 for stable numbers.
    params = sum(p.numel() for p in net.parameters())
    flops = get_flops(net, imgsz=imgsz)

    if half and dev.type == "cuda":
        net = net.half()
    dtype = torch.float16 if half and dev.type == "cuda" else torch.float32
    x = torch.randn(batch, 3, imgsz, imgsz, device=dev, dtype=dtype)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = net(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            _ = net(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms_per_image = 1000.0 * dt / iters / batch
    fps = 1000.0 / ms_per_image

    result = dict(
        model=model_path,
        device=str(dev),
        imgsz=imgsz,
        batch=batch,
        half=bool(half and dev.type == "cuda"),
        params_M=params / 1e6,
        flops_G=float(flops),
        latency_ms=ms_per_image,
        fps=fps,
    )
    return result


def print_result(r: dict) -> None:
    print(f"model         : {r['model']}")
    print(f"device        : {r['device']}")
    print(f"imgsz         : {r['imgsz']}")
    print(f"batch         : {r['batch']}")
    print(f"half (fp16)   : {r['half']}")
    print(f"params (M)    : {r['params_M']:.3f}")
    print(f"FLOPs (G)     : {r['flops_G']:.2f}")
    print(f"latency (ms)  : {r['latency_ms']:.2f}")
    print(f"FPS           : {r['fps']:.1f}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latency / FPS benchmark for a YOLO model")
    p.add_argument("model", help="path to .yaml or .pt model")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="0", help="'0' / 'cpu'")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--half", action="store_true", help="fp16 forward (GPU only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    r = benchmark(args.model, args.imgsz, args.batch, args.device, args.warmup, args.iters, args.half)
    print_result(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
