from ultralytics import YOLO
from config import YAML_PATH, WEIGHTS_PATH, IMAGE_SIZE

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}. Set YOLO_WEIGHTS or train first.")

model = YOLO(str(WEIGHTS_PATH))

metrics = model.val(
    data=str(YAML_PATH),
    imgsz=IMAGE_SIZE,
    split="val"
)

for metric_name, value in metrics.results_dict.items():
    print(f"{metric_name}: {value}")
