from ultralytics import YOLO

best_model = "/home/mohammad/AdvanceML/runs/detect/my_project/visdrone_person_car_1280/weights/best.pt"

model = YOLO(best_model)

metrics = model.val(
    data="/home/mohammad/datasets/visdrone.yaml",
    imgsz=1280,
    split="val"
)

for metric_name, value in metrics.results_dict.items():
    print(f"{metric_name}: {value}")
