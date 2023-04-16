from collections import namedtuple
from functools import reduce
from enum import Enum

import numpy as np
import cv2

from src.utils import get_angle_diff, weighted_avg, calc_euler_angles
from src.params import VisionParams

class OrientMethod(Enum):
    BEST_REF = 0
    WEIGHT_AVG = 1
    RECOVER_POSE = 2

class OrientationFinder:
    """
    Finds the orientation/side in the field based on the background.
    Has references images with known orientation and compares them
    to a new image to estimate the orientation in which the image.
    was taken.
    """    
    Reference = namedtuple('Reference', ['img', 'angle', 'points', 'descriptor'])
    RefMatch = namedtuple('RefMatch', ['ref_angle', 'num_matches'])

    def __init__(self, ref_imgs, ref_angles, vision_params:VisionParams, intrinsic_mtx=None) -> None:
        """
        Initializes the Orientation Finder
        :param ref_imgs: List with the reference images
        :param ref_angles: List with the angles of each reference image, in the same order as the images
        """
        self.params = vision_params

        self.intrinsic_mtx = intrinsic_mtx

        self.detector = cv2.ORB_create(
            nfeatures=self.params.nfeatures, scaleFactor=self.params.scaleFactor,
            patchSize=self.params.patchSize, edgeThreshold=self.params.patchSize
        )
        self.matcher = cv2.FlannBasedMatcher(
            indexParams={ 'algorithm':6, 'table_number':6, 'key_size':12, 'multi_probe_level':1},
            searchParams={'checks': self.params.checks}
        )

        self.references = [
            self.Reference(ref_img, ref_angles[i], *self.detector.detectAndCompute(ref_img, None))
            for i, ref_img in enumerate(ref_imgs)
        ]

    def get_num_equal_pts(self, ref, img_descriptors):
        """
        Returns the number of points that strongly match with the given
        image and a reference image.
        :param ref: Reference image to count the number of equal points.
        Has the descriptors and points.
        :param img_descriptors: descriptors for the image points found by the detector.
        :return: number of equal/matched points between the two images.
        """
        matches = self.matcher.knnMatch(ref.descriptor, img_descriptors, k=2)
        # We count the number of strong matches using a heuristic distance factor
        num_equal_pts = reduce(
            lambda val, match:
                val + (1 if len(match) >=2
                            and match[0].distance < self.params.dist_ratio_thres*match[1].distance
                         else 0),
            matches, 0
        )
        return num_equal_pts

    def get_equal_pts(self, ref, img_pts, img_descriptors):
        """
        Returns the number of points that match with the given
        image and a reference image.
        :param ref: Reference image to count the number of equal points.
        Has the descriptors and points.
        :param img_pts: image points found by the detector.
        :param img_descriptors: descriptors for the image points found by the detector.
        :return: np.array with the matched points position in the reference and in the image.
        """
        matches = self.matcher.knnMatch(ref.descriptor, img_descriptors, k=2)
        # We determine the strong_matches using a heuristic distance factor
        strong_matches = [
            match[0] for match in matches
            if len(match) >= 2 and match[0].distance < self.params.dist_ratio_thres*match[1].distance
        ]
        equal_ref_pts = np.array([ref.points[r.queryIdx].pt for r in strong_matches], dtype=np.float32)
        equal_img_pts = np.array([img_pts[r.trainIdx].pt for r in strong_matches], dtype=np.float32)
        return equal_ref_pts, equal_img_pts

    def get_best_ref(self, img_descriptors):
        """
        Returns the best reference found.
        :return: Best reference found
        """
        best_ref = max(
            self.references, key=lambda ref: self.get_num_equal_pts(ref, img_descriptors)
        )
        return best_ref

    def calc_orientation_best_ref(self, img_descriptors):
        """
        Calculates the orientation according to the best reference found.
        :return: Orientation angle in degrees. Limited to [0, 360[
        """
        return self.get_best_ref(img_descriptors).angle

    def calc_orientation_weight_avg(self, img_descriptors):
        """
        Calculates the orientation according to the best three references found.
        It estimates the orientation angle to be the weighted average of those angles
        with the weights being the number of matches.
        :param img_descriptors: descriptors for the image points found by the detector.
        :return: Orientation angle in degrees. Limited to [0, 360[
        """
        # List of tuples storing the number of matches for each reference
        # The first entry is the reference angle and the second the number of matches
        ref_matches_list: list[self.RefMatch] = []

        for ref in self.references:
            points_quantity = self.get_num_equal_pts(ref, img_descriptors)
            ref_matches_list.append(self.RefMatch(ref.angle, points_quantity))

        ref_matches_list.sort(key=lambda c: c.num_matches, reverse=True)
        main_angle = ref_matches_list[0].ref_angle

        values = [
            get_angle_diff(main_angle, ref_matches_list[i].ref_angle)
            for i in range(min(len(ref_matches_list), 3))
        ]
        weights = [
            ref_matches_list[i].num_matches
            for i in range(min(len(ref_matches_list), 3))
        ]

        delta_angle = weighted_avg(values, weights)
        angle = main_angle + delta_angle
        return angle if angle >= 0 else 360 + angle

    def calc_orientation_recover_pose(self, img_pts, img_descriptors):
        """
        Calculates the orientation according to the cv2.recoverPose function.
        :param img_pts: image points found by the detector.
        :param img_descriptors: descriptors for the image points found by the detector.
        :return: Orientation angle in degrees. Limited to [0, 360[
        """
        ref = self.get_best_ref(img_descriptors)
        equal_ref_pts, equal_img_pts = self.get_equal_pts(ref, img_pts, img_descriptors)
        try:
            _, _, rotation_mtx, translation_versor, inliers = cv2.recoverPose(
                points1=equal_ref_pts, points2=equal_img_pts, cameraMatrix1=self.intrinsic_mtx,
                distCoeffs1=None, cameraMatrix2=self.intrinsic_mtx, distCoeffs2=None,
                method=cv2.USAC_ACCURATE, prob=self.params.prob, threshold=self.params.threshold
            )
            # Robot's yaw is camera's pitch
            _, delta_pitch, _ = calc_euler_angles(rotation_mtx)
            return (ref.angle + delta_pitch)%360
        except cv2.error:
            # Hack: for some reason, on the reference images, opencv uses the wrong
            # overloaded function and it raises an assertion error.
            return ref.angle

    def calc_orientation(self, img,  method=OrientMethod.RECOVER_POSE):
        """
        Calculates the orientation using the desired mode.
        :param img: Image in which the orientation is to be calculated
        :param method: Method in which to estimate the orientation
        :return: Orientation angle in degrees. Limited to [0, 360[
        """

        img_pts, img_descriptors = self.detector.detectAndCompute(img, None)

        if  method == OrientMethod.BEST_REF:
            return self.calc_orientation_best_ref(img_descriptors)
        elif  method == OrientMethod.WEIGHT_AVG:
            return self.calc_orientation_weight_avg(img_descriptors)
        elif  method == OrientMethod.RECOVER_POSE:
            return self.calc_orientation_recover_pose(img_pts, img_descriptors)
