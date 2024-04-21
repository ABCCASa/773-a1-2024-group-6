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
    score = None
    H = None

    while iter_count > 0 or H is None:
        iter_count -= 1
        sample_idx = np.random.choice(pair_count, size=4, replace=False)
        samples = np_points[sample_idx]
        if valid_quadrangle(samples[0][0], samples[1][0], samples[2][0], samples[3][0], 8):
            tempH = calculate_transformation(samples)
            temp_score = calculate_score(np_points, tempH, distance_threshold)
            if H is None or temp_score > score:
                H = tempH
                score = temp_score
    return H, score


def valid_triangle(p1, p2, p3, min_area):
    areaOfTriangle = 0.5 * (p1[0] * (p2[1]-p3[1]) + p2[0] * (p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    return abs(areaOfTriangle) >= min_area


def valid_quadrangle(p1, p2, p3, p4, min_area):
    if not valid_triangle(p1, p2, p3, min_area):
        return False
    if not valid_triangle(p1, p2, p4, min_area):
        return False
    if not valid_triangle(p1, p3, p4, min_area):
        return False
    if not valid_triangle(p2, p3, p4, min_area):
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
    result = np.dot(points, H.T)
    result = result[:, :2] / result[:, 2, None]
    return result


def calculate_score(point_pairs, H, max_distance):
    left_points = point_pairs[:, 0]
    right_points = point_pairs[:, 1, :2]
    transformed_point = transformation_points(left_points, H)
    distances = np.sqrt(np.sum((transformed_point-right_points)**2, axis=1))
    return np.where(distances <= max_distance)[0].shape[0]
