# IntelRobotChallangeGigglers

Team Gigglers – Intel Physical AI Challenge (Junction 2026): SO-101 arm + Anomalib/OpenVINO
brick inspection cell.

- `so101_jog.py` – keyboard jogging of the SO-101 follower (Windows-only, uses `msvcrt`).
- `vision/` – anomaly detection (capture, train, predict): see [vision/README.md](vision/README.md).
- `dataset/` – shared photos for training/testing: see [dataset/README.md](dataset/README.md).
- `collection/` – robot data collector (all cameras + depth + arm pose): see [collection/README.md](collection/README.md).
- `docs/` – guides and notes: **[docs/vision_pipeline.md](docs/vision_pipeline.md) (start here)**, **[docs/dataset_workflow.md](docs/dataset_workflow.md) (collecting a dataset)**, **[docs/station_runbook.md](docs/station_runbook.md) (OpenVINO on the Intel laptop)**, [docs/inspection_plan_2cam.md](docs/inspection_plan_2cam.md), [docs/experiments.md](docs/experiments.md), [docs/photo-guide.md](docs/photo-guide.md).
