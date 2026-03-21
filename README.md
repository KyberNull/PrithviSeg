> Develop an AI/ML model to identify key features from SVAMITVA Scheme drone orthophotos.

# Scope
- Extract building footprints and classify rooftops (RCC, Tiled, Tin, etc.).
- Extract road networks and waterbodies.
- Identify infrastructure locations such as distribution transformers, overhead tanks, wells.
- Optimize the model for efficient processing and deployment.

## Project Structure
- `train_phase_1.py`: Phase 1 pretraining on SBD segmentation (`NUM_CLASSES=21`).
- `train_phase_2.py`: Phase 2 pretraining on LoveDA (`NUM_CLASSES=7`).
- `train.py`: Phase 3 fine-tuning on the geospatial target dataset (`NUM_CLASSES=4`).
- `evaluate.py`: Evaluation/visualization utility (VOC-style validation).
- `model.pt`: Shared checkpoint file used across phases.

## Environment Setup
This repo uses `uv` and `pyproject.toml` (no `.env`-driven runtime configuration in current code).

```bash
uv sync
source .venv/bin/activate
```

## Training Workflow
Run phases in order:

```bash
PYTHON_GIL=1 uv run train_phase_1.py
PYTHON_GIL=1 uv run train_phase_2.py
PYTHON_GIL=1 uv run train.py
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
PYTHON_GIL=1 uv run evaluate.py
```

`evaluate.py` reports `mCEL` and `mIoU`, and can display qualitative predictions.

## Data Note
For VOC-style masks, label value `255` means ignore region. During plotting, mask ignored pixels if you want class colors to be visually consistent.
