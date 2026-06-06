yaml_content = """
path: /home/mohammad/datasets

train: VisDrone2019-DET-train/VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/images

nc: 2

names:
  0: person
  1: car
"""

yaml_path = "/home/mohammad/datasets/visdrone.yaml"

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("YAML CREATED:", yaml_path)
