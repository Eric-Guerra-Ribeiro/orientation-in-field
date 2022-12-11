from pathlib import Path

import cv2
import numpy as np

from src.orientation_finder import OrientationFinder, OrientMethod
from src.utils import get_angle_diff, build_intrinsic_mtx

# Camera Params
fov = 1.012300
cx = 320
cy = 240
fx = cx/np.tan(fov/2)
fy = cy/np.tan(fov/2)

intrinsic_mtx = build_intrinsic_mtx(fx, fy, cx, cy)

dataset_path = Path("./dataset/A/")

use45s = False

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

orientation_finder = OrientationFinder(ref_imgs, ref_angles, intrinsic_mtx)
angle_diff_vec = []

for i in range(1):
    angle_diff_vec.append(abs(get_angle_diff(angles[i],  orientation_finder.calc_orientation(imgs[i], OrientMethod.RECOVER_POSE))))

print ("Mean: " + str(np.array(angle_diff_vec).mean()))
print ("Std Dev: " + str(np.array(angle_diff_vec).std()))
