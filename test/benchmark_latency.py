"""Latency (ms/slice) and peak RAM per backend.
python benchmark_latency.py                       # random slices, every model found in weights/
python benchmark_latency.py --t1ce a.nii.gz --flair b.nii.gz   # stream a real volume (end-to-end RAM)"""
import argparse
import glob
import os
import threading
import time

import numpy as np
import psutil

from segmenter import Segmenter
from slicestream import SliceStream

ap = argparse.ArgumentParser()
ap.add_argument("--weights", nargs="*"); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--t1ce"); ap.add_argument("--flair")
a = ap.parse_args()

proc, peak, stop = psutil.Process(os.getpid()), [0], threading.Event()


def sampler():
    while not stop.is_set():
        peak[0] = max(peak[0], proc.memory_info().rss)
        time.sleep(0.01)


def slices():
    if a.t1ce and a.flair:
        for _, raw in SliceStream.from_paths(a.t1ce, a.flair):
            yield raw
    else:
        rng = np.random.default_rng(0)
        for _ in range(a.n + a.warmup):
            yield np.abs(rng.normal(300, 100, (2, 240, 240))).astype(np.float32)


paths = a.weights or sorted(glob.glob("weights/*.onnx") + glob.glob("weights/*.pt")) or [None]
print(f"{'backend':16}{'mean ms':>9}{'p50':>8}{'p95':>8}{'slices/s':>10}{'peak RAM MB':>13}")
for p in paths:
    seg = Segmenter(weights_path=p)
    peak[0] = 0; stop.clear(); threading.Thread(target=sampler, daemon=True).start()
    times = []
    for i, raw in enumerate(slices()):
        t0 = time.perf_counter(); seg.predict_mask(raw); dt = (time.perf_counter() - t0) * 1000
        if i >= a.warmup:
            times.append(dt)
    stop.set(); time.sleep(0.05)
    t = np.array(times)
    print(f"{seg.backend:16}{t.mean():9.1f}{np.percentile(t,50):8.1f}{np.percentile(t,95):8.1f}"
          f"{1000/t.mean():10.1f}{peak[0]/1e6:13.0f}")
print("Note: peak RAM is whole-process RSS (includes Python + library baseline). Target: < 2048 MB.")
