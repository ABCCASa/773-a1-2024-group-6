import math
import numpy as np
from ImageStitching.fitting import transformation_points, transformation_point

from matplotlib import pyplot
import colorsys
import cv2
import utils



def warping(base_image, match_images):
    image_size, offset = check_image_size(base_image, match_images)
    offset_x, offset_y = offset

    result_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    result_image[offset_x:base_image.shape[0] + offset_x, offset_y:base_image.shape[1] + offset_y] = base_image

    points_idx = []
    for r in range(image_size[0]):
        for c in range(image_size[1]):
            points_idx.append([r, c, 1])

    transfer_points_all_images = []
    np_points = np.array(points_idx)
    np_points[:, 0] -= offset_x
    np_points[:, 1] -= offset_y
    for image, H in match_images:
        transfer_points = transformation_points(np_points, H)
        transfer_points_all_images.append(transfer_points)

    np_index = np.array(points_idx)
    for i in range(len(match_images)):
        transfer_points = transfer_points_all_images[i]
        overlap_image = match_images[i][0]
        vaild_area = np.where((0<=transfer_points[:,0])
                                &(transfer_points[:,0]< overlap_image.shape[0] - 1)
                                &(0 <=transfer_points[:,1])
                                &(transfer_points[:,1]< overlap_image.shape[1] - 1))
        colors = colors_mix(overlap_image, transfer_points[vaild_area])
        valid_np_index = np_index[vaild_area]
        result_image[valid_np_index[:,0], valid_np_index[:,1]] = colors

    result_image = result_image.astype(np.uint8)

    return result_image




def check_image_size(base_image, match_images):
    base_shape = base_image.shape
    images_corners = []
    corners_base = np.array([
        [0, 0],
        [base_shape[0], 0],
        [0, base_shape[1]],
        [base_shape[0], base_shape[1]]
    ])
    images_corners.append(corners_base)

    for image, H in match_images:
        image_shape = image.shape
        H = np.linalg.inv(H)
        corners = np.array([
            [0, 0, 1],
            [image_shape[0], 0, 1],
            [0, image_shape[1], 1],
            [image_shape[0], image_shape[1], 1]
        ])
        corners = transformation_points(corners, H)
        images_corners.append(corners)

    corners = np.concatenate(images_corners, axis=0)
    min_x = math.floor(np.min(corners[:, 0]))
    min_y = math.floor(np.min(corners[:, 1]))
    max_x = math.ceil(np.max(corners[:, 0]))
    max_y = math.ceil(np.max(corners[:, 1]))
    offset = (-min_x, -min_y)
    image_size = (max_x-min_x, max_y-min_y)
    return image_size, offset


def colors_mix(image, points):
    quotient, remainder = np.divmod(points, 1)
    quotient = quotient.astype(int)
    a = remainder[:,0]
    b = remainder[:,1]
    x = quotient[:,0]
    y = quotient[:,1]
    return image[x, y]* ((1-a) * (1-b))[:, None]\
                + image[x, y+1] * (a * (1-b))[:, None]\
                + image[x+1, y] * (b * (1-a))[:, None]\
                + image[x+1, y+1] * (a * b)[:, None]


def rgb_to_hsv(color):
    r = color[0]/255.0
    g = color[1]/255.0
    b = color[2]/255.0
    return np.array(colorsys.rgb_to_hsv(r, g, b))


def hsv_to_rgb(color):
    h = color[0]
    s = color[1]
    v = color[2]
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    r = int(r*255)
    g = int(g*255)
    b = int(b*255)
    return np.array([r, g, b])

