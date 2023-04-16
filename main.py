from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from src.orientation_finder import OrientationFinder, OrientMethod
from src.utils import get_angle_diff, build_intrinsic_mtx
from src.params import VisionParams

# Camera Params
fov = 1.012300
cx = 320
cy = 240
fx = cx/np.tan(fov/2)
fy = cy/np.tan(fov/2)

intrinsic_mtx = build_intrinsic_mtx(fx, fy, cx, cy)

# dataset_path = Path("./dataset/irl/")
# image_path = Path("./dataset/irl/test2_210_test.png")

dataset_path = Path("./dataset/sepulchral/")
image_path = Path("./dataset/sepulchral/test4_15_test.png")
use45s = False

ref_imgs = []
ref_angles = []

imgs = []
angles = []
test_cases = []

for file in dataset_path.glob("*.png"):
    split_name = file.name.split("_")
    img_angle = int(split_name[1])
    img_case = split_name[0]
    img = cv2.imread(str(file), cv2.IMREAD_ANYCOLOR)
    if img_case == "ref":
        if use45s or img_angle in {0, 90, 180, 360}:
            ref_imgs.append(img)
            ref_angles.append(img_angle)
    imgs.append(img)
    angles.append(img_angle)
    test_cases.append(img_case)

vision_params = VisionParams(1230, 1.8847186328577767, 44, 18, 0.9827160798859101, 7)

orientation_finder = OrientationFinder(
    ref_imgs, ref_angles, vision_params, intrinsic_mtx
)
errors = []

### Comment the lines below to select the method to use
# orient_method = OrientMethod.BEST_REF
# orient_method = OrientMethod.WEIGHT_AVG
orient_method = OrientMethod.RECOVER_POSE

img = cv2.imread(str(image_path))

angle = orientation_finder.calc_orientation(img, orient_method)

print(angle)

# times = []
# for i in range(len(imgs)):
#     start_time = time.time()
#     errors.append(abs(get_angle_diff(angles[i],  )))
#     end_time = time.time()
#     times.append(end_time - start_time)
#     break

# times = 1000*np.array(times) # Converting to ms

# print(f"Execution Time: {times.mean():.0f}±{times.std():.0f} ms per iteration")
# print ("Mean: %.2f degrees" % np.array(errors).mean())
# print ("Std Dev: %.2f degrees" % np.array(errors).std())    

# plt.hist(errors, bins=np.linspace(0, 50, 25), histtype='bar', ec='black')
# plt.title('Error Histogram')
# plt.xlabel('Angle Error')
# plt.ylabel('Frequency')
# plt.locator_params(axis='y', integer=True)
# plt.show()
