import numpy as np


def projective_transformation_fitting(point_pairs, distance_threshold=1, iter_count=1000):
    pair_count = len(point_pairs)
    # convert (x, y) to (row, column, 1)
    preprocess_points = []
    for p1, p2 in point_pairs:
        p1 = [p1[1], p1[0], 1]
        p2 = [p2[1], p2[0], 1]
        preprocess_points.append([p1, p2])

    # to np
    np_points = np.array(preprocess_points)

    # result ready to return
    inliers_count = 0
    H = None

    while iter_count > 0 or H is None:
        iter_count -= 1
        sample_idx = np.random.choice(pair_count, size=4, replace=False)
        samples = np_points[sample_idx]
        if valid_quadrangle(samples[0][0], samples[1][0], samples[2][0], samples[3][0]):
            tempH = calculate_transformation(samples)
            temp_inliers_count = get_inliers(np_points, tempH, distance_threshold).shape[0]
            if H is None or temp_inliers_count > inliers_count:
                H = tempH
                inliers_count = temp_inliers_count

    # get inliers and use them to calculate the final transformation
    inliers = get_inliers(np_points, H, distance_threshold)
    H = calculate_transformation(inliers)

    return H, inliers_count


def valid_triangle(p1, p2, p3):
    areaOfTriangle = 0.5 * (p1[0] * (p2[1]-p3[1]) + p2[0] * (p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    return abs(areaOfTriangle) >= 1e-5


def valid_quadrangle(p1, p2, p3, p4):
    if not valid_triangle(p1, p2, p3):
        return False
    if not valid_triangle(p1, p2, p4):
        return False
    if not valid_triangle(p1, p3, p4):
        return False
    if not valid_triangle(p2, p3, p4):
        return False
    return True


def calculate_transformation(point_pairs):
    A = []
    for p1, p2 in point_pairs:
        A.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]])
        A.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]])
    A = np.array(A)
    U, D, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H


def transformation_point(point, H):
    result = H@point
    result /= result[2]
    return result[:2]


def transformation_points(points, H):
    result = np.einsum('ij,nj->ni', H, points)
    result = result[:, :2] / result[:, 2, None]
    return result


def get_inliers(point_pairs, H, max_distance):
    left_points = point_pairs[:, 0]
    right_points = point_pairs[:, 1, :2]
    transformed_point = transformation_points(left_points, H)
    distances = np.sum((transformed_point-right_points)**2, axis=1)
    return point_pairs[distances <= max_distance**2]

