from pathlib import Path
import time

import numpy as np
import cv2

from src.params import VisionParams
from src.utils import build_intrinsic_mtx, get_angle_diff
from src.orientation_finder import OrientationFinder, OrientMethod

class Tester:
    dataset_path = Path("./dataset/")
    folders = [
        "jbhcentral", "kiara", "paul_lobe_haus",
        "sepulchral", "shangai", "stadium", "ulm"
    ]
    fov = 1.012300
    cx = 320
    cy = 240
    fx = cx/np.tan(fov/2)
    fy = cy/np.tan(fov/2)

    intrinsic_mtx = build_intrinsic_mtx(fx, fy, cx, cy)

    def __init__(self, params:VisionParams, orient_method:OrientMethod, ref_angles:set()):
        self.params = params
        self.orient_method = orient_method
        self.ref_angles = ref_angles
    
    def performance(self):
        times = []
        errors = []
        for folder in self.folders:
            path = self.dataset_path / folder

            ref_imgs = []
            ref_angles = []

            imgs = []
            angles = []
            test_cases = []

            for file in path.glob("*.png"):
                split_name = file.name.split(".")
                split_name = split_name[0].split("_")
                img_case = split_name[0]
                img_angle = int(split_name[1])
                img_test_train = split_name[2]
                img = cv2.imread(str(file), cv2.IMREAD_ANYCOLOR)
                if img_case == "ref" and (img_angle in self.ref_angles):
                    ref_imgs.append(img)
                    ref_angles.append(img_angle)
                if img_test_train == "train":
                    imgs.append(img)
                    angles.append(img_angle)
                    test_cases.append(img_case)
            
            orientation_finder = OrientationFinder(
                ref_imgs, ref_angles, self.params, self.intrinsic_mtx
            )

            for i in range(len(imgs)):
                start_time = time.time()
                errors.append(
                    abs(get_angle_diff(
                        angles[i],  orientation_finder.calc_orientation(imgs[i], self.orient_method)
                    ))
                )
                end_time = time.time()
                times.append(end_time - start_time)
        times = 1000*np.array(times)
        errors = np.array(errors)
        return times.mean(), times.std(), errors.mean(), errors.std()
