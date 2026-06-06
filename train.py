from ultralytics import YOLO
from config import YAML_PATH, MODEL_NAME, PROJECT_NAME, RUN_NAME

model = YOLO(MODEL_NAME)

history = model.train(
    data=YAML_PATH,
    epochs=100,
    batch=12,
    optimizer="AdamW",
    lr0=0.0008,
    lrf=0.01,

    project=PROJECT_NAME,
    name="visdrone_person_car_1280",

    patience=25,
    imgsz=1280,

    mosaic=1.0,
    close_mosaic=10,
    mixup=0.05,
    copy_paste=0.2,

    scale=0.7,
    translate=0.15,
    fliplr=0.5,

    degrees=5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    workers=0,
    cache=False,

    save=True,
    plots=True,
)

print("Training Finished")
