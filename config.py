import os
from pathlib import Path

DATASET_ROOT = Path(os.environ.get("VISDRONE_ROOT", Path(__file__).parent / "data"))

TRAIN_IMAGES = DATASET_ROOT / "VisDrone2019-DET-train/VisDrone2019-DET-train/images"
TRAIN_LABELS = DATASET_ROOT / "VisDrone2019-DET-train/VisDrone2019-DET-train/labels"

VALID_IMAGES = DATASET_ROOT / "VisDrone2019-DET-val/VisDrone2019-DET-val/images"
VALID_LABELS = DATASET_ROOT / "VisDrone2019-DET-val/VisDrone2019-DET-val/labels"

TEST_IMAGES = DATASET_ROOT / "VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/images"
TEST_LABELS = DATASET_ROOT / "VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/labels"

YAML_PATH = DATASET_ROOT / "visdrone.yaml"

MODEL_NAME = "yolov10b.pt"

PROJECT_NAME = "my_project"
RUN_NAME = "visdrone_person_car"
IMAGE_SIZE = int(os.environ.get("YOLO_IMGSZ", "1280"))
WEIGHTS_PATH = Path(os.environ.get(
    "YOLO_WEIGHTS",
    Path(PROJECT_NAME) / "visdrone_person_car_1280/weights/best.pt",
))
