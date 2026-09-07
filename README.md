# Drone Person and Car Detection with YOLOv10

This project fine-tunes Ultralytics **YOLOv10b** (the balanced variant) to detect pedestrians and cars in VisDrone2019 aerial imagery. It includes VisDrone-to-YOLO annotation conversion, dataset configuration, training, validation, inference and visualization scripts.

## Dataset pipeline

Only two VisDrone classes are retained: class 1 `pedestrian` becomes YOLO class 0 `person`, and class 4 `car` becomes YOLO class 1 `car`. Conversion normalizes bounding boxes by image dimensions and skips ignored/out-of-scope records.

`VisDrone annotations → two-class YOLO labels → dataset YAML → YOLOv10b transfer learning → validation/inference`

## Training configuration

The committed code loads `yolov10b.pt`, trains for up to 100 epochs with batch size 12, AdamW, early-stopping patience 25 and augmentation. The report describes experiments at 640, 960 and 1280 pixels.

## Reported results

| Resolution | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| 640 | 0.689 | 0.521 | 0.544 | 0.310 |
| 960 | 0.742 | 0.601 | 0.652 | 0.396 |
| 1280 | **0.806** | **0.743** | **0.806** | **0.519** |

For the reported 1280 run, class mAP@50 is 0.900 for cars and 0.713 for people. These values come from `finalReport.pdf`; the trained weights and full 960/1280 run directories are not committed, so those results cannot be reproduced from a fresh clone without retraining. A committed notebook output identifies YOLOv10b with 19,005,654 parameters and 91.6 GFLOPs.

## Setup and run

Download the VisDrone2019 DET train, validation and test-dev archives, preserving their nested directory names.

```bash
python -m venv .venv
source .venv/bin/activate
# Install a CUDA-compatible PyTorch build first when using a GPU.
pip install -r requirements.txt
export VISDRONE_ROOT=/absolute/path/to/datasets

python dataset/convert_annotations.py
python dataset/yaml_creator.py
python dataset/class_counter.py
python train.py
python evaluate.py
python inference.py
```

## Structure

- `config.py` — paths and run identifiers
- `dataset/` — annotation conversion, YAML creation and class counts
- `train.py`, `evaluate.py`, `inference.py` — Ultralytics entry points
- `visualization/plot_samples.py` — converted-box inspection
- `visdrone-yolov10-object-detection.ipynb` — experiment notebook
- `visdrone-yolov10-object-detection.py` — historical notebook export retained as experiment evidence; it is not a maintained setup entry point and contains machine-specific paths
- `finalReport.pdf` — methodology and reported metrics

## Known limitations

- VisDrone data and trained weights are not included.
- The validation split is used for model selection and reported evaluation; there is no independent labeled test evaluation.
- The 960/1280 result artifacts are absent, so report values are not independently verifiable here.
- The notebook and its historical `.py` export retain environment-specific experiment paths. Use only the maintained scripts listed in Setup with `VISDRONE_ROOT` for the portable workflow.
- There are no automated tests, CI, deployment files or production-serving implementation.
