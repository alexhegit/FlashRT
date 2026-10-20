"""Synthetic XR-1 smoke/latency check; does not measure robot task success."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import flash_rt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--attention', choices=['sdpa', 'eager'], default='sdpa')
    parser.add_argument('--num-steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10, help='Measured calls after warmup')
    parser.add_argument('--native', action='store_true', help='Use FlashRT HIP DiT ops')
    parser.add_argument('--output', default='results/xr1/smoke.json')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    start = time.perf_counter()
    model = flash_rt.load_model(
        checkpoint=args.checkpoint, config='xr1', framework='torch',
        hardware='amd_rdna35', use_fp8=False, num_views=3,
        num_steps=args.num_steps, attention=args.attention, fuse=args.native, autotune=0)
    load_ms = (time.perf_counter() - start) * 1000
    rng = np.random.default_rng(42)
    images = [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(3)]
    state = np.zeros(60 if model.pipeline.foundation else 8, np.float32)
    expected_shape = (30, 60) if model.pipeline.foundation else (10, 7)
    torch.cuda.synchronize()
    start = time.perf_counter()
    first = model.predict(images, 'pick up the red block', state)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - start) * 1000
    latencies = []
    repeat_max_abs = 0.0
    for _ in range(args.runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        actions = model.predict(images, state=state)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        if actions.shape != expected_shape or not np.isfinite(actions).all():
            raise RuntimeError(f'Invalid XR-1 actions: {actions.shape}')
        repeat_max_abs = max(repeat_max_abs, float(np.max(np.abs(actions - first))))
    result = dict(
        torch=torch.__version__, hip=torch.version.hip, native=args.native,
        arch=torch.cuda.get_device_properties(0).gcnArchName,
        checkpoint=str(Path(args.checkpoint).resolve()),
        attention=args.attention, num_steps=args.num_steps, shape=list(actions.shape),
        finite=True, repeat_max_abs=repeat_max_abs, load_ms=load_ms,
        first_call_ms=first_call_ms, latency_ms=latencies,
        median_ms=float(np.median(latencies)), p95_ms=float(np.percentile(latencies, 95)),
        aotriton_experimental=os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '0'),
        synthetic_observations=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
