# Station runbook — OpenVINO optimisation on the Intel Core Ultra laptop

Goal: the numbers for the judging criterion "OpenVINO & Intel Core Ultra optimisation" (latency,
throughput, deployment efficiency, appropriate use of NPU / GPU / CPU) and "system efficiency"
(latency, thermal headroom, power) — in one run. Everything below runs from the repo root.

## 1. Setup (once, ~10 min)

```bash
git fetch && git checkout vision && git pull          # or the branch with these scripts
bash scripts/station_setup.sh
```

Creates `.venv`, installs CPU-only torch (the Intel GPU / NPU are used through OpenVINO) and
`requirements.txt`, lists the OpenVINO devices. Expect `CPU`, `GPU`, `NPU`. If GPU / NPU are missing:
drivers, and your user in the `render` group (`sudo usermod -aG render $USER`, log out / in).
For power numbers (RAPL): `sudo chmod -R a+r /sys/class/powercap/intel-rapl:0`.

## 2. Models

With the clean dataset collected (`docs/dataset_workflow.md`):

```bash
python vision/build_views.py
python vision/train_views.py --int8          # FP16 + INT8 per model, registry results/models/models.json
python vision/train_face_classifier.py       # classifier with preprocessing inside the OpenVINO graph
```

Speed does not depend on model quality, so any trained registry (even from old data) gives valid
optimisation numbers.

## 3. Run the check

```bash
bash scripts/station_run.sh                  # ~2-5 min
```

Writes `results/station/report.md` (+ `station.json`, `bench.csv`):

| section | what it shows | criterion |
|---|---|---|
| system | CPU, OpenVINO version, devices with full names, idle power / temperature | context |
| models × devices × precision | compiles or not (unsupported ops reported), compile time cold / with cache, latency p50 / p95, FPS, mJ per inference | latency, throughput, device choice, precision |
| everything at once | Anomalib on NPU + classifier on GPU + a proxy load on GPU (stand-in for the VLA policy): latency alone vs together | heterogeneous use of NPU / GPU / CPU |
| control loop | 100 Hz loop on the CPU (robot stand-in): timing jitter idle vs under full AI load | "robot control stays reliable" |
| power / temperature | package W under load, temperature before / after | system efficiency |
| recommended device map | fastest working device per model; NPU for Anomalib when within 1.5× (lowest power, GPU stays free for VLA) | device choice, with a reason |

Options: `--runs 200`, `--load-seconds 20`, `--anomaly-device NPU --classifier-device GPU --policy-device GPU`.

## 4. Use the result

- Put the chosen map into the inspector:

  ```python
  InspectorBank("results/models/models.json", mode="per_face", anomaly_device="NPU", classifier_device="GPU")
  ```

  Each model compiles with fallback (NPU → HETERO:NPU,CPU → GPU → CPU), uses the model cache
  (`results/ov_cache`, so later starts are fast) and is warmed up at start; `bank.devices` shows
  where each model really runs.
- If PatchCore does not compile on the NPU (its memory-bank search), the table says so; options:
  `HETERO:NPU,CPU` (picked automatically), PaDiM on the NPU, or PatchCore on the GPU.
- Ask the VLA team for their policy's OpenVINO export + device + rate (Hz) to complete the device map.

## What is already optimised (for the pitch)

- Anomalib → OpenVINO IR with a static input `[1, 3, 256, 256]` (fixes Anomalib 2.7's batch=2 export
  bug; static shapes are required for the NPU), FP16 and INT8 (NNCF post-training quantisation).
- Face classifier → OpenVINO with **preprocessing inside the graph** (PrePostProcessor: BGR→RGB,
  float, normalise, layout) — less Python, runs on the same device as the model.
- Model cache + latency hint + warm-up: no multi-second NPU compile at the start of the demo.
- Device fallback: an unsupported device never crashes the demo; it is visible in the logs.
- Early exit per brick (judges' rule): defective bricks need fewer inspections → shorter cycle.
- Still CPU-bound: brick segmentation (~1 s per frame with the cluttered-background method) →
  fixed booth window + plain backdrop (`paper` mode) on the clean data is the next latency win.
