from collections import namedtuple
import numpy as np
import PIL.Image
import cv2

Reference = namedtuple('Reference', ['img', 'angle', 'points', 'descriptor'])

class OrientationFinder:
    """
    Finds the orientation/side in the field based on the background

    Has references images with known orientation and compares them
    to a new image to estimate the orientation in which the image
    was taken.
    """
    
    def __init__(self, ref_imgs, ref_angles) -> None:
        """
        Initializes the Orientation Findes
        :param ref_imgs: List with the reference images
        :param ref_angles: List with the angles of each reference image, in the same order as the images
        """
        self.detector = cv2.ORB_create(nfeatures=4500, scaleFactor=1.19)
        self.matcher = cv2.FlannBasedMatcher(
            indexParams={ 'algorithm':6, 'table_number':6, 'key_size':12, 'multi_probe_level':1},
            searchParams={'checks': 50}
        )
        self.references = [
            Reference(ref_img, ref_angles[i], *self.detector.detectAndCompute(ref_img, None))
            for i, ref_img in enumerate(ref_imgs)
        ]

    def get_equal_points(self, ref, img):
        """
        Returns the number of points that match with the given
        image and a reference image.        
        :param ref_img: reference image to count the number of
        equal points.
        :return: number of equal points between the two images.
        """
        img_points, img_descriptors = self.detector.detectAndCompute(img, None)
        matches = self.matcher.knnMatch(ref.descriptor, img_descriptors, k=2)
        # We determine the strong_matches using a heuristic distance factor
        strong_matches = [r1 for r1, r2 in matches if r1.distance < 0.7 * r2.distance]
        equal_ref_pts = np.array([ref.points[r.queryIdx].pt for r in strong_matches], dtype=np.float32)
        equal_img_pts = np.array([img_points[r.queryIdx].pt for r in strong_matches], dtype=np.float32)
        return equal_ref_pts, equal_img_pts

    def draw_lines_difference(self, ref_pts, img_pts, img):
        
        # img = np.repeat(img[:, :, None], 3, axis=-1)
        for p, pref in zip(img_pts, ref_pts): # Desenha uma linha ligando os pares de pontos
            print(p)
            print(pref)
            cv2.line(img, np.uint(pref), np.uint(p), (250, 0, 0))
        PIL.Image.fromarray(img).show()

    def eval_image(self, img):
        """
        Prints the closest orientation that match the current image.
        """

        def get_angle_diff(angle1, angle2):
            phi = abs(angle2-angle1) % 360
            sign = 1
            # Used to calculate sign
            if not ((angle1-angle2 >= 0 and angle1-angle2 <= 180) or (
                    angle1-angle2 <= -180 and angle1-angle2 >= -360)):
                sign = -1
            if phi > 180:
                result = 360-phi
            else:
                result = phi

            return result*sign

        correspondece_vector = []

        best_ref_pts = None
        best_img_pts = None
        most_pts = -1

        for ref in self.references:
            ref_pts, img_pts = self.get_equal_points(ref, img)
            points_quantity = len(ref_pts)
            if points_quantity > most_pts:
                best_ref_pts = ref_pts
                best_img_pts = img_pts
                most_pts = points_quantity
            correspondece_vector.append((ref.angle, points_quantity))
        correspondece_vector.sort(key=lambda c: -c[1])



        self.draw_lines_difference(ref_pts, img_pts, img)

        main_angle = int(correspondece_vector[0][0])
        
        num = get_angle_diff(main_angle, int(correspondece_vector[1][0])) * correspondece_vector[1][1] + get_angle_diff(main_angle, int(correspondece_vector[2][0])) * correspondece_vector[2][1]
        den = correspondece_vector[0][1] + correspondece_vector[1][1] + correspondece_vector[2][1]
        angle = main_angle + num/den
        if angle < 0:
            angle = 360 + angle
        return angle