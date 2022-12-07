from pathlib import Path

import cv2

from src.orientation_finder import OrientationFinder

dataset_path = Path("./dataset/A/")

ref_imgs = []
ref_angles = []

imgs = []
angles = []
test_cases = []

for file in dataset_path.glob("*.png"):
    split_name = file.name.split("_")
    img_angle = split_name[-1].split(".")[0]
    img_case = split_name[0]
    img = cv2.imread(str(file), cv2.IMREAD_ANYCOLOR)
    if img_case == "ref":
        ref_imgs.append(img)
        ref_angles.append(img_angle)
    imgs.append(img)
    angles.append(img_angle)
    test_cases.append(img_case)

orientation_finder = OrientationFinder(ref_imgs, ref_angles)
