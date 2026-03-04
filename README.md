
> Develop an AI/ML model to identify key features from SVAMITVA Scheme drone orthophotos.

# Scope
- Extract building footprints and classify rooftops (RCC, Tiled, Tin, etc.).
- Extract road networks and waterbodies.
- Identify infrastructure locations such as distribution transformers, overhead tanks, wells.
- Achieve a minimum 95% accuracy.
- Optimize the model for efficient processing and deployment.

## Input Data
- Drone imagery for 10 villages (training & validation).
- Drone imagery for 10 additional villages (output testing).

## Expected Deliverables
- A fully trained and optimized AI model for orthophoto feature extraction.
- Feature-extracted datasets for training villages.
- Technical documentation covering model design, training, and deployment.
- Final report with accuracy metrics and improvement recommendations.

> Note: For VOC Segmentation, mask image has a special pixel value 255 that means “ignore this area.” `plt.imshow` treats that 255 as the top of the color scale. Real classes are only 0 to 20, so they got squeezed into very dark colors. That’s why you mostly saw the boundary/outline and not full regions.

## Per-device configuration (`.env`)

The project now supports device-specific configuration through a local `.env` file.

1. Copy `.env.example` to `.env`.
2. Set values that vary by machine (for example `TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`, and `NUM_WORKERS`).
3. Run training/evaluation as usual; values are loaded automatically.

Available keys:

- `MODEL_PATH`
- `NUM_CLASSES`
- `NUM_WORKERS`
- `TRAIN_BATCH_SIZE`
- `TRAIN_LEARNING_RATE`
- `TRAIN_WEIGHT_DECAY`
- `TRAIN_EPOCHS`
- `EVAL_BATCH_SIZE`
- `EVAL_MAX_EXAMPLES`
- `IGNORE_LABEL`
