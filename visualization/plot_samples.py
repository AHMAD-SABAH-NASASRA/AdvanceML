import os
import cv2
import matplotlib.pyplot as plt

train_image = "/home/mohammad/datasets/VisDrone2019-DET-train/VisDrone2019-DET-train/images"

train_label = "/home/mohammad/datasets/VisDrone2019-DET-train/VisDrone2019-DET-train/labels"

class_names = [
    "person",
    "car",
]

image_files = sorted(os.listdir(train_image))[:25]

plt.figure(figsize=(20, 20))

for idx, img_file in enumerate(image_files, 1):

    img_path = os.path.join(train_image, img_file)

    label_path = os.path.join(
        train_label,
        img_file.replace(".jpg", ".txt")
    )

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    if os.path.exists(label_path):

        with open(label_path) as f:

            for line in f:

                cls, x, y, bw, bh = map(float, line.split())

                cls = int(cls)

                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)

                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    img,
                    class_names[cls],
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

    plt.subplot(5, 5, idx)

    plt.imshow(img)

    plt.axis("off")

plt.tight_layout()

plt.show()
