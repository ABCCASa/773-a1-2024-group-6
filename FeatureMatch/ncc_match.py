import numpy as np
import math


def NCC(img1, img2):
    img1_sub_mean = img1 - np.mean(img1)
    img2_sub_mean = img2 - np.mean(img2)
    return np.sum(img1_sub_mean * img2_sub_mean) / math.sqrt(np.sum(img1_sub_mean**2)*np.sum(img2_sub_mean**2))


# get window from image
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
    min_value = window_size//2
    max_x = img.shape[1]-min_value-1
    max_y = img.shape[0] - min_value - 1
    return [point for point in points if min_value <= point[0] <= max_x and min_value < point[1] <= max_y]


# use matrix to parallel calculate the NCC of all window
def NCCMatrix(features1, features2):
    shape1 = features1.shape
    count1 = shape1[0]
    count2 = features2.shape[0]
    feature_matrix1 = np.repeat(features1, count2, axis=0).reshape(shape1[0], count2, shape1[1], shape1[2])
    feature_matrix2 = np.tile(features2, (count1, 1, 1, 1))

    features1_mean = np.mean(feature_matrix1, axis=(2, 3))
    features2_mean = np.mean(feature_matrix2, axis=(2, 3))

    features_sub_mean_1 = feature_matrix1 - features1_mean[:, :, None, None]
    features_sub_mean_2 = feature_matrix2 - features2_mean[:, :, None, None]

    top = np.sum(features_sub_mean_1 * features_sub_mean_2, axis=(2, 3))
    bottom = np.sqrt(np.sum(features_sub_mean_1**2, axis=(2, 3)) * np.sum(features_sub_mean_2**2, axis=(2, 3)))
    return top / bottom


def NCCMatch(img1, points1, img2, points2, window_size=15, ratio_threshold=0.9, two_dir_match=False):
    points1 = point_clamp(img1, points1, window_size)
    points2 = point_clamp(img2, points2, window_size)

    features1 = get_features(img1, points1, window_size)
    features2 = get_features(img2, points2, window_size)
    score_matrix = NCCMatrix(features1, features2)

    score_matrix = (score_matrix+1) / 2
    matched_points = []

    for i in range(score_matrix.shape[0]):
        scores = score_matrix[i]
        sorted_arg = np.argsort(scores)
        j = sorted_arg[-1]
        ratio = scores[sorted_arg[-2]] / scores[j]
        if ratio < ratio_threshold:
            matched_points.append((points1[i], points2[j]))

    if two_dir_match:
        for i in range(score_matrix.shape[1]):
            scores = score_matrix[:, i]
            sorted_arg = np.argsort(scores)
            j = sorted_arg[-1]
            ratio = scores[sorted_arg[-2]] / scores[j]
            if ratio < ratio_threshold:
                matched_points.append((points1[j], points2[i]))
    return matched_points
