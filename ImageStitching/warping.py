import math

import numpy as np
from ImageStitching.fitting import transformation_points, transformation_point
from matplotlib import pyplot


def warping(left_image, right_image, H):
    image_size, offset = check_board_size(left_image.shape, right_image.shape, H)
    result_image = np.zeros((image_size[0], image_size[1], 3),dtype=int)

    result_image[offset[0]:left_image.shape[0]+offset[0], offset[1]:left_image.shape[1]+offset[1]] = left_image

    for r in range(result_image.shape[0]):
        for c in range(result_image.shape[1]):
            point = transformation_point(np.array([r-offset[0], c-offset[1], 1], dtype=float), H)
            r2, c2 = int(point[0]), int(point[1])
            if 0 <= r2 < right_image.shape[0] and 0 <= c2 < right_image.shape[1]:
                result_image[r, c] = right_image[r2, c2]

    pyplot.imshow(result_image, cmap='gray')
    pyplot.show()


def check_board_size(shape0, shape1, H):
    H = np.linalg.inv(H)
    corners_right = np.array([
        [0, 0, 1],
        [shape1[0], 0, 1],
        [0, shape1[1], 1],
        [shape1[0], shape1[1], 1]
    ], dtype=float)

    corners_right = transformation_points(corners_right, H)
    corners_left = np.array([
        [0, 0],
        [shape0[0], 0],
        [0, shape0[1]],
        [shape0[0], shape0[1]]
    ], dtype=float)

    corner = np.concatenate((corners_left, corners_right), axis=0)

    min_x = math.floor(np.min(corner[:, 0]))
    min_y = math.floor(np.min(corner[:, 1]))
    max_x = math.ceil(np.max(corner[:, 0]))
    max_y = math.ceil(np.max(corner[:, 1]))
    offset = (-min_x, -min_y)
    image_size = (max_x-min_x, max_y-min_y)
    return image_size, offset









