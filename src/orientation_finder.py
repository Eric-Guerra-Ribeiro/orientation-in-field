from collections import namedtuple
from functools import reduce
from enum import Enum

import numpy as np
import cv2

from src.utils import get_angle_diff, weighted_avg

class OrientMode(Enum):
    BEST_REF = 0
    WEIGHT_AVG = 1


class OrientationFinder:
    """
    Finds the orientation/side in the field based on the background

    Has references images with known orientation and compares them
    to a new image to estimate the orientation in which the image
    was taken.
    """
    
    Reference = namedtuple('Reference', ['img', 'angle', 'points', 'descriptor'])
    RefMatch = namedtuple('RefMatch', ['ref_angle', 'num_matches'])

    def __init__(self, ref_imgs, ref_angles) -> None:
        """
        Initializes the Orientation Finder
        :param ref_imgs: List with the reference images
        :param ref_angles: List with the angles of each reference image, in the same order as the images
        """
        self.detector = cv2.ORB_create(nfeatures=4500, scaleFactor=1.19)
        self.matcher = cv2.FlannBasedMatcher(
            indexParams={ 'algorithm':6, 'table_number':6, 'key_size':12, 'multi_probe_level':1},
            searchParams={'checks': 50}
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
            lambda val, match: val + (1 if match[0].distance < 0.7*match[1].distance else 0),
            matches, 0
        )
        return num_equal_pts

    def get_equal_pts(self, ref, img_descriptors, img_pts):
        """
        Returns the number of points that match with the given
        image and a reference image.
        :param ref: Reference image to count the number of equal points.
        Has the descriptors and points.
        :param img_descriptors: descriptors for the image points found by the detector.
        :param img_pts: image points found by the detector.
        :return: np.array with the matched points position in the reference and in the image.
        """
        matches = self.matcher.knnMatch(ref.descriptor, img_descriptors, k=2)
        # We determine the strong_matches using a heuristic distance factor
        strong_matches = [r1 for r1, r2 in matches if r1.distance < 0.7 * r2.distance]
        equal_ref_pts = np.array([ref.points[r.queryIdx].pt for r in strong_matches], dtype=np.float32)
        equal_img_pts = np.array([img_pts[r.trainIdx].pt for r in strong_matches], dtype=np.float32)
        return equal_ref_pts, equal_img_pts

    def calc_orientation_best_ref(self, num_matches_list):
        """
        Calculates the orientation according to the best reference found.
        :param num_matches_list: List with the number of matches for each reference
        :return: Orientation angle in degrees. Limited to [0, 360[
        """
        angle = max(num_matches_list, key=lambda c: c.num_matches).ref_angle
        return angle

    def calc_orientation_weight_avg(self, num_matches_list):
        """
        Calculates the orientation according to the best three references found.
        It estimates the orientation angle to be the weighted average of those angles
        with the weights being the number of matches.
        :param num_matches_list: List with the number of matches for each reference
        :return: Orientation angle in degrees. Limited to [0, 360[
        """
        num_matches_list.sort(key=lambda c: c.num_matches, reverse=True)
        main_angle = num_matches_list[0].ref_angle
        
        values = [
            get_angle_diff(main_angle, num_matches_list[i].ref_angle)
            for i in range(min(len(num_matches_list), 3))
        ]
        weights = [
            num_matches_list[i].num_matches
            for i in range(min(len(num_matches_list), 3))
        ]

        delta_angle = weighted_avg(values, weights)
        angle = main_angle + delta_angle
        return angle if angle >= 0 else 360 + angle


    def calc_orientation(self, img, mode=OrientMode.WEIGHT_AVG):
        """
        Calculates the orientation using the desired mode.
        :param img: Image in which the orientation is to be calculated
        :param mode: Mode in which to estimate the orientation
        :return: Orientation angle in degrees. Limited to [0, 360[
        """

        img_points, img_descriptors = self.detector.detectAndCompute(img, None)

        # List of tuples storing the number of matches for each reference
        # The first entry is the reference angle and the second the number of matches
        num_matches_list: list[self.RefMatch] = []

        for ref in self.references:
            points_quantity = self.get_num_equal_pts(ref, img_descriptors)
            num_matches_list.append(self.RefMatch(ref.angle, points_quantity))
        
        if mode == OrientMode.BEST_REF:
            return self.calc_orientation_best_ref(num_matches_list)
        elif mode == OrientMode.WEIGHT_AVG:
            return self.calc_orientation_weight_avg(num_matches_list)
