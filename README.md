> Develop an AI/ML model to identify key features from SVAMITVA Scheme drone orthophotos.

# Scope
- Extract building footprints and classify rooftops (RCC, Tiled, Tin, etc.).
- Extract road networks and waterbodies.
- Identify infrastructure locations such as distribution transformers, overhead tanks, wells.
- Optimize the model for efficient processing and deployment.

 # 🛰️ MOPR Geospatial Hackathon: Semantic Segmentation from Drone Imagery

 <p align="center">
	 <img src="https://img.shields.io/badge/python-3.14+-blue.svg" alt="Python Version">
	 <img src="https://img.shields.io/badge/Model-SegFormer-orange.svg" alt="Model">
 </p>

 ## 🚀 Overview

 This project develops a deep learning pipeline for **semantic segmentation** of drone orthophotos, focusing on extracting building footprints, classifying rooftops, mapping road networks, and identifying key infrastructure for the [SVAMITVA Scheme](https://svamitva.nic.in/). The solution is optimized for efficient processing and deployment on large-scale geospatial datasets.


 ## ✨ Key Features

 - **Multi-phase Training:** Pretraining on LoveDA, fine-tuning on geospatial targets
 - **SegFormer Backbone:** Modern transformer-based semantic segmentation
 - **Flexible Dataset Support:** Easily extendable for new geospatial datasets
 - **Robust Evaluation:** VOC-style validation and qualitative visualization
 - **Efficient Checkpointing:** Smart resume and phase transition logic


 ## 🗂️ Project Structure

 ```text
 ├── train.py           # Phase 3: Fine-tuning on geospatial dataset
 ├── pretrain.py        # Phase 2: Pretraining on LoveDA
 ├── evaluate.py        # Evaluation and visualization
 ├── model.py           # SegFormer model definition
 ├── datasets.py        # Dataset loading utilities
 ├── losses.py          # Loss functions and metrics
 ├── transforms.py      # Data augmentation and transforms
 ├── utils.py           # Utility functions
 ├── model.pt           # Shared model checkpoint
 ├── data/              # Data directory (see below)
 └── ...
 ```

 **Data Directory Layout:**
 ```text
 data/
	 phase-2/   # LoveDA dataset
	 phase-3/   # Target geospatial dataset
 ```


 ## ⚙️ Installation & Environment Setup

 This project uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.

 ```bash
 uv sync
 source .venv/bin/activate
 ```


 ## 🏋️ Training & Evaluation Workflow

 Run each phase in order for best results:

 ```bash
 # Phase 1: Pretrain on LoveDA
 uv run pretrain.py

 # Phase 2: Fine-tune on geospatial dataset
 uv run train.py

 # Evaluate
 uv run evaluate.py
 ```

 Or use the provided VS Code tasks: **Pretrain**, **Train**, **Evaluate**.


 ## 🧠 Model Details

 - **Architecture:** [SegFormer](https://arxiv.org/abs/2105.15203) (MiT-b2 backbone, via `segmentation_models_pytorch`)
 - **Losses:** Dice, Focal, IoU
 - **Metrics:** mIoU, per-class IoU, pixel accuracy
 - **Checkpoints:** All phases read/write `model.pt` (auto-handles class-count transitions)


 ## 📊 Results & Visualizations

 \[WIP\] Example outputs and validation metrics will be shown here after training.

 ## 🙏 Acknowledgements

 - [SVAMITVA Scheme](https://svamitva.nic.in/)
 - [SegFormer Paper](https://arxiv.org/abs/2105.15203)
 - [LoveDA Dataset](https://github.com/Junjue-Wang/LoveDA)
 - [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Environment Setup
This repo uses `uv` and `pyproject.toml` (no `.env`-driven runtime configuration in current code).

```bash
uv sync
```

## Training Workflow
Run phases in order:

```bash
uv run pretrain.py
uv run train.py
```

Equivalent VS Code tasks are also available: `Pretrain`, `Train`, and `Evaluate`.

## Checkpoint and Resume Semantics
- All phases read/write `model.pt`.
- On class-count mismatch (for example 21->7 or 7->4), segmentation head weights are dropped and training state is reset for a clean phase transition.
- Optimizer/scheduler/scaler states are resumed only for true in-phase continuation.
- This prevents mixed optimizer/scheduler state and unintended LR carry-over across phases.

## Notes on Stability
- Ignore label `255` is used where applicable (`CrossEntropyLoss(ignore_index=255)` in phase 2).
- If loss becomes `NaN`, do not continue from that checkpoint; restart from the latest known-good checkpoint (or phase boundary) with fresh optimizer/scheduler state.
- Monitor LR in logs; abrupt spikes usually indicate resume-state mismatch or schedule misalignment.

## Evaluation

```bash
uv run evaluate.py
```

`evaluate.py` reports `mCEL` and `mIoU`, and can display qualitative predictions.

## Data Note
For VOC-style masks, label value `255` means ignore region. During plotting, mask ignored pixels if you want class colors to be visually consistent.
