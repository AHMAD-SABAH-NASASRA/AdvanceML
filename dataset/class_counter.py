import os
from collections import Counter

CLASS_NAMES = {
    0: "person",
    1: "car",
}

def count_classes(label_dir):

    counts = Counter()

    for file in os.listdir(label_dir):

        if not file.endswith(".txt"):
            continue

        with open(os.path.join(label_dir, file)) as f:

            for line in f:

                values = line.strip().split()

                if len(values) != 5:
                    continue

                cls = int(values[0])

                counts[cls] += 1

    return counts


if __name__ == "__main__":

    train_label = "/home/mohammad/datasets/VisDrone2019-DET-train/VisDrone2019-DET-train/labels"

    counts = count_classes(train_label)

    for k, v in counts.items():
        print(CLASS_NAMES[k], v)
