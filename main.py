from pathlib import Path

import cv2
import numpy as np

from src.orientation_finder import OrientationFinder
from src.utils import get_angle_diff

dataset_path = Path("./dataset/A/")

use45s = True

ref_imgs = []
ref_angles = []

imgs = []
angles = []
test_cases = []

for file in dataset_path.glob("*.png"):
    split_name = file.name.split("_")
    img_angle = int(split_name[-1].split(".")[0])
    img_case = split_name[0]
    img = cv2.imread(str(file), cv2.IMREAD_ANYCOLOR)
    if img_case == "ref":
        if use45s or img_angle in {0, 90, 180, 360}:
            ref_imgs.append(img)
            ref_angles.append(img_angle)
    imgs.append(img)
    angles.append(img_angle)
    test_cases.append(img_case)

orientation_finder = OrientationFinder(ref_imgs, ref_angles)
angle_diff_vec = []

for i in range(len(imgs)):
    angle_diff_vec.append(abs(get_angle_diff(angles[i],  orientation_finder.calc_orientation(imgs[i]))))

print ("Mean: " + str(np.array(angle_diff_vec).mean()))
print ("Std Dev: " + str(np.array(angle_diff_vec).std()))
