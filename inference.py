from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
from config import TEST_IMAGES, WEIGHTS_PATH, IMAGE_SIZE

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}. Set YOLO_WEIGHTS or train first.")

model = YOLO(str(WEIGHTS_PATH))

test_images_path = str(TEST_IMAGES)

all_images = sorted(os.listdir(test_images_path))[:15]

for img_name in all_images:

    img_path = os.path.join(test_images_path, img_name)

    results = model.predict(
        source=img_path,
        imgsz=IMAGE_SIZE,
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
