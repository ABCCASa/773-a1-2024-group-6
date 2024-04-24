import numpy as np
def get_feature(img, point, window_size):
    row = point[1]
    column = point[0]
    offset = window_size//2
    return img[row-offset: row+offset, column-offset:column+offset]


# get all windows from image
def get_features(img, points, window_size):
    features = []
    for point in points:
        feature = get_feature(img, point, window_size)
        features.append(feature)
    return np.array(features)


# remove the point out size the boundary
def point_clamp(img, points, window_size):
    min_value =window_size//2
    max_x = img.shape[1]-min_value-1
    max_y = img.shape[0]-min_value - 1
    return [point for point in points if min_value <= point[0] <= max_x and min_value < point[1] <= max_y]

def SSDAMatrix(features1, features2):
    shape1 = features1.shape
    count1 = shape1[0]
    count2 = features2.shape[0]

    feature_matrix1 = np.repeat(features1, count2, axis=0).reshape(shape1[0], count2, shape1[1], shape1[2])
    feature_matrix2 = np.tile(features2, (count1, 1, 1, 1))

    features1_mean = np.mean(feature_matrix1, axis=(2, 3))
    features2_mean = np.mean(feature_matrix2, axis=(2, 3))

    features_sub_mean_1 = feature_matrix1 - features1_mean[:, :, None, None]
    features_sub_mean_2 = feature_matrix2 - features2_mean[:, :, None, None]

    SSDA = abs(features_sub_mean_1 - features_sub_mean_2)
    sum_SSDA = np.sum(SSDA, axis=(2, 3))
    return sum_SSDA
def SSDAMatch(img1, points1, img2, points2, window_size=15):
    points1 = point_clamp(img1, points1, window_size)
    points2 = point_clamp(img2, points2, window_size)

    features1 = get_features(img1, points1, window_size)
    features2 = get_features(img2, points2, window_size)

    score_matrix = SSDAMatrix(features1, features2)
    matched_points = []

    for i in range(score_matrix.shape[0]):
        scores = score_matrix[i]
        sorted_arg = np.argsort(scores)
        j = sorted_arg[0]
        ratio = scores[j]/scores[sorted_arg[1]]
        if ratio < 0.8:
            matched_points.append((points1[i], points2[j]))
    return matched_points
