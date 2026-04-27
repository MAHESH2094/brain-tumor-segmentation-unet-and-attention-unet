# Brain Tumor Segmentation with U-Net and Attention U-Net

End-to-end BraTS brain tumor segmentation pipeline using U-Net, Attention U-Net, and an optional ViT branch, with reproducible training, evaluation, and export workflows.

## Overview

This project segments brain tumors from 4-channel MRI scans: FLAIR, T1, T1ce, and T2.
The codebase is organized so the full workflow can be reproduced, evaluated, and exported from normal Python entrypoints instead of notebook-only cells.

## Results

### Cell 10 evaluation summary

| Model | Dice (soft) | Dice (hard) | IoU | Precision | Recall | F1 | Pixel Accuracy | Test Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7626 | 0.7625 | 0.7141 | 0.9676 | 0.7406 | 0.7634 | 0.9845 | 0.1662 |
| Attention U-Net | 0.7373 | 0.8160 | 0.7594 | 0.8405 | 0.8968 | 0.8225 | 0.9901 | 0.0581 |

### Architecture comparison

| Metric | U-Net | Attention U-Net |
|---|---:|---:|
| Total parameters | 20,563,265 | 21,262,469 |
| Number of layers | 69 | 97 |
| Attention gates | 0 | 4 |
| Parameter overhead | - | +3.40% |

Attention U-Net keeps the capacity increase small while focusing the skip connections on tumor-relevant features.
In this evaluation, it improves hard Dice, IoU, recall, F1, pixel accuracy, and test loss, which makes it the stronger clinical tradeoff even though U-Net keeps slightly higher precision and soft Dice.

## Pipeline

- `environment.py`: seed setup, TensorFlow runtime setup, and smoke checks
- `config.py`: shared configuration and output paths
- `dataset_paths.py`, `dataset_analysis.py`, `dataset_builder.py`: dataset discovery, inspection, and HDF5 creation
- `preprocessing.py`: normalization and image preparation
- `augmentation.py`: training-time augmentation
- `model_blocks.py`, `unet.py`, `attention_unet.py`, `attention_unet_vit.py`: model components and architectures
- `metrics.py`: loss functions and segmentation metrics
- `train.py`: model training and threshold generation
- `evaluate.py`: evaluation and comparison table generation
- `inference.py`: single-patient or batch inference
- `export.py`: packaged run outputs and final report artifacts
- `upload_predict.py`: final upload-to-prediction flow

## Key Insight

Attention U-Net adds 4 attention gates and only +3.40% parameters over U-Net.
It keeps precision lower than U-Net, but it pushes recall from 0.7406 to 0.8968 and hard Dice from 0.7625 to 0.8160.
That is the better tradeoff here because missing tumor tissue is more costly than a small precision drop.

## Tech Stack

- TensorFlow / Keras
- NumPy
- HDF5 / h5py
- NiBabel
- OpenCV
- Matplotlib
- scikit-learn
- tqdm
- Pillow

## How to Run

```bash
pip install -r requirements.txt
python environment.py
python train.py
python evaluate.py
python export.py
python upload_predict.py
```

If you are running in Kaggle or another notebook environment, use `optimal_config_kaggle.py` or `optimal_config_kaggle_ultrafast.py` as the runtime config source.

## Project Structure

```text
.
├── environment.py
├── config.py
├── dataset_paths.py
├── dataset_analysis.py
├── preprocessing.py
├── dataset_builder.py
├── augmentation.py
├── model_blocks.py
├── unet.py
├── attention_unet.py
├── attention_unet_vit.py
├── metrics.py
├── train.py
├── evaluate.py
├── inference.py
├── export.py
├── upload_predict.py
├── pipeline.py
├── ml_pipeline/
├── models/
├── results/
└── export/
```

## Future Work

- Add 1 to 2 prediction-vs-ground-truth figures from the latest inference export
- Run a fresh full U-Net + Attention U-Net comparison export and refresh the summary table
- Try 3D model variants and stronger loss functions
