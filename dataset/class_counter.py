import os
from collections import Counter
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import TRAIN_LABELS

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

    counts = count_classes(TRAIN_LABELS)

    for k, v in counts.items():
        print(CLASS_NAMES[k], v)
