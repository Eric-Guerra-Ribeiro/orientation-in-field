from functools import reduce

def get_angle_diff(angle1, angle2):
    """
    Calculates the signed difference bewteen two angles.
    :param angle1: first angle in degrees.
    :param angle2: second angle in degrees.
    :return: angle1 - angle2, limited to ]-180, 180]
    """
    phi = angle2-angle1
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
