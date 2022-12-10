from functools import reduce

import numpy as np

def get_angle_diff(angle1, angle2):
    """
    Calculates the signed difference bewteen two angles.
    :param angle1: first angle in degrees.
    :param angle2: second angle in degrees.
    :return: angle1 - angle2, limited to ]-180, 180]
    """
    phi = angle2 - angle1
    # Calculating the sign
    sign = 1
    if not ((-180 <= phi <= 0) or (180 <= phi <= 360)):
        sign = -1
    # Limiting phi
    phi = abs(phi)%360
    phi = 360 - phi if phi > 180 else phi
    return sign*phi


def weighted_avg(values, weigths):
    """
    Calculates the weighted average of the values.
    :param values: list of values.
    :param weights: list of weights in the same order as the list of values.
    """
    sum_weights = sum(weigths)
    avg = reduce(
        lambda cumm_sum, item: cumm_sum + item[1]*item[0], zip(values, weigths), 0
    )/sum_weights
    return avg


def build_intrinsic_mtx(fx, fy, cx, cy):
    """
    Returns the intrinsic matrix.
    :param fx: Focal Length on x axis.
    :param fy: Focal Length on y axis.
    :param cx: Center on x axis.
    :param cy: Center on y axis.
    :return: Intrinsic matrix as np.array.
    """
    return np.array([[fx, 0., cx],
                     [0., fy, cy],
                     [0., 0., 1.]])


def calc_euler_angles(rotation_mtx):
    """
    Calculates the Euler angles: yaw, pitch and roll.
    :param rotation_mtx: Rotation matrix.
    :return: Yaw, Pitch and Roll in degrees.
    """
    sy = np.sqrt(rotation_mtx[0][0]**2 +  rotation_mtx[1][0]**2)
    singular = sy < 1e-6
    if not singular:
        yaw = np.arctan2(rotation_mtx[1][0], rotation_mtx[0][0])
        pitch = np.arctan2(-rotation_mtx[2][0], sy)
        roll = np.arctan2(rotation_mtx[2][1] , rotation_mtx[2][2])
    else:
        yaw = 0.
        pitch = np.arctan2(-rotation_mtx[2][0], sy)
        roll = np.arctan2(-rotation_mtx[1][2], rotation_mtx[1][1])
    return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)
