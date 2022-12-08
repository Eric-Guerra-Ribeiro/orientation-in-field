def get_angle_diff(angle1, angle2):
    """
    Calculates the signed difference bewteen two angles
    :param angle1: first angle in degrees
    :param angle2: second angle in degrees
    :return: angle1 - angle2, limited in ]-180, 180]
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
