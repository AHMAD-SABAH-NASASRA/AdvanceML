from pathlib import Path
from PIL import Image
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATASET_ROOT

CLASS_MAP = {
    1: 0,
    4: 1,
}

def convert_split(split):

    root = DATASET_ROOT / split / split

    ann_dir = root / "annotations"
    img_dir = root / "images"
    label_dir = root / "labels"

    label_dir.mkdir(exist_ok=True)

    for ann_file in ann_dir.glob("*.txt"):

        img_file = img_dir / (ann_file.stem + ".jpg")

        if not img_file.exists():
            continue

        w, h = Image.open(img_file).size

        out = []

        for line in ann_file.read_text().splitlines():

            vals = line.split(",")

            if len(vals) < 6:
                continue

            x, y, bw, bh = map(float, vals[:4])

            ignore = int(vals[4])

            cls = int(vals[5])

            if ignore == 0:
                continue

            if cls not in CLASS_MAP:
                continue

            new_cls = CLASS_MAP[cls]

            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h

            out.append(f"{new_cls} {xc} {yc} {nw} {nh}")

        (label_dir / ann_file.name).write_text("\n".join(out))


for split in [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test-dev",
]:
    convert_split(split)

print("DONE")
