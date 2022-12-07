from collections import namedtuple

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
