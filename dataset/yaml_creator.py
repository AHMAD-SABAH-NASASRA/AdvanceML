import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATASET_ROOT, YAML_PATH

yaml_content = f"""
path: {DATASET_ROOT}

train: VisDrone2019-DET-train/VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/images

nc: 2

names:
  0: person
  1: car
"""

DATASET_ROOT.mkdir(parents=True, exist_ok=True)
with open(YAML_PATH, "w") as f:
    f.write(yaml_content)

print("YAML CREATED:", YAML_PATH)
