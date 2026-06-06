from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

best_model = "/home/mohammad/AdvanceML/runs/detect/my_project/visdrone_person_car_1280/weights/best.pt"

model = YOLO(best_model)

test_images_path = "/home/mohammad/datasets/VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/images"

all_images = sorted(os.listdir(test_images_path))[:15]

for img_name in all_images:

    img_path = os.path.join(test_images_path, img_name)

    results = model.predict(
        source=img_path,
        imgsz=1280,
        conf=0.15,
        save=True
    )

    annotated = results[0].plot()

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(15, 8))
    plt.imshow(annotated)
    plt.title(img_name)
    plt.axis("off")
    plt.show()