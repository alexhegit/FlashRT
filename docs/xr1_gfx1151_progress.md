# XR-1 gfx1151 implementation progress

Updated 2026-10-21 00:17, Asia/Shanghai. Branch: `feat/xr1-gfx1151`.

Implemented and GPU validated:

- `flash_rt.load_model(config="xr1", framework="torch", hardware="amd_rdna35", num_views=3)`.
- Published XR-1 5B `model_states.pt`: all 1135 tensors load strictly;
  output `(30,60)`, quantile state normalization, upstream camera ordering,
  resize and prompt, action normalization, and asynchronous action prefixes.
- RoboCasa deploy checkpoint: output `(10,7)`, upstream custom processor,
  and changing state while reusing the instruction.
- BF16 reference mode without FlashRT extensions or `flash-attn`.
- Opt-in native HIP DiT GEMM/RMSNorm using existing FlashRT kernels.
  Vision Conv3d retains MIOpen. No new kernel/CMake ABI changes.
- Independent Python 3.12 / ROCm 7.2 environment, setup/build script, pinned
  downloads, normalization preparation, synthetic benchmark and real-video
  validation scripts. See [deployment guide](xr1_gfx1151.md).

Final implementation measurements on this gfx1151 device, PyTorch ROCm 2.10.0,
Transformers 4.57.1, five denoise steps, and experimental AOTriton enabled:

| Input / checkpoint | Reference median | Native median / p95 |
|---|---:|---:|
| Synthetic 3x256x256 / RoboCasa | 435.12 ms | 360.90 / 365.33 ms |
| Synthetic 3x256x256 / 5B | 453.12 ms | 377.52 / 383.86 ms |
| Real demo camera crops / 5B | 707.17 ms | 652.80 ms median |

Synthetic results use ten measured calls after first-call warmup. The real
input medians aggregate three measured calls at each of twelve frames from
`XiaomiRobotics/xr1_post_train_demo/json1` (0 through 440, step 40). Timers
include preprocessing, model execution and action download; model loading
and first-call initialization are recorded separately.

All twelve reference outputs exactly matched the original upstream runtime
message/resize/state utilities and direct generation on the same loaded ROCm
policy, including upstream action decoding. Native real-input action cosine
was at least 0.999851; maximum absolute difference was 0.031529. Fixed-seed
predictions are deterministic and preserve the caller's CPU/GPU RNG. These
checks establish numerical agreement under the tested inputs, not robot
control limits or task success.

A vision patch GEMM optimization was removed after real video exposed larger
BF16 errors than synthetic inputs. The final benchmarks above use retained
MIOpen vision projection and replace the earlier experimental latency figures.

Validation:

- 107 XR-1/API/RDNA tests passed, including both real-checkpoint E2E cases,
  async prefixes, and all twelve real-camera observations.
- The standalone demo validator passed all twelve upstream/reference/native
  comparisons using the final implementation.
- `pip check`, Python/shell syntax checks, and `git diff --check` passed.
- Wheel built and imported from outside the source tree; vendored sources and
  license are included, without compiled binaries or test files.

Final local evidence under `results/xr1/`:

- `final_tests.txt`
- `robocasa_reference_final.json`, `robocasa_native_final.json`
- `5b_reference_final.json`, `5b_native_final.json`
- `demo_validation_final.json` and matching `.npz` observations
- `real_upstream_parity.json`, `wheel_build_final.txt`

Weights, environments, videos and generated evidence remain local and are
excluded from Git.

Practical limits:

- The 5B smoke checkpoint uses upstream `load_washer` demonstration statistics.
  Deployment requires the target embodiment's statistics and post-training.
- No robot/simulation task-success benchmark was run and no robot was controlled.
- Native BF16 is opt-in (`fuse=True`); reference mode is the default. Attention
  remains ROCm SDPA. Full-model graph capture is not implemented.
- Hardware qualification covers gfx1151 only.

The one-shot user timer `flashrt-xr1-morning-report-v2.timer` is active for
2026-10-21 08:00:00 Asia/Shanghai, accuracy 1 second. It invokes `codex queue`
from this project directory for the current conversation. Queue insertion was
verified; autonomous delivery at 08:00 has not yet occurred and depends on
this machine and Codex session processing remaining available. At that time,
report actual results without starting new modifications or downloads because
the user's temporary broad authorization expires then.
