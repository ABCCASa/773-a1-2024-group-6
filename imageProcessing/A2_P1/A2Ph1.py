
from A2_P1.utilities import *
import utils
import A2_P1.A2_utilities as A2_utilities
import math
from A2_P1.Get3d2dCoordinates import *
from A2_P1.plot_image import plot_image_with_point_group, plot_image_with_lines

def apply_kappa(corners, center, k1, k2=None):
    delta = corners - center
    distance = 1 + np.sum(delta ** 2, axis=1) * k1
    if k2 is not None:
        distance += np.sum(delta ** 4, axis=1) * k2
    return delta * distance[:, None] + center


# 3. Define the RANSAC line-fitting algorithm to get the best slope and intercept of each horizontal line of points
def distance_to_line(points, a, b):
    return np.abs(a * points[:, 0] - points[:, 1] + b) / math.sqrt(a ** 2 + 1)

def get_inlier(corners, idxes, threshold=2):
    p1, p2 = corners[idxes[0]], corners[idxes[1]]
        # y = ax + b
    if p2[0] - p1[0] == 0:
        return None, None, -10000, 100000
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p2[1] - a * p2[0]
    distances = distance_to_line(corners, a, b)
    return a, b, len(np.where(distances <= threshold)), np.sum(distances ** 2)

def RANSAC(corners):
    count = corners.shape[0]
    max_inlier = -1

    best_a, best_b, min_error= None, None, 1e20
    for j in range(count):
        for k in range(j + 1, count):
            current_a, current_b, current_inlier, current_error = get_inlier(corners, [j, k])
            if current_inlier > max_inlier or current_error < min_error:
                best_a, best_b, min_error = current_a, current_b, current_error
                max_inlier = current_inlier
            elif current_inlier == max_inlier and current_error < min_error:
                best_a, best_b, min_error = current_a, current_b, current_error
                max_inlier = current_inlier
    return best_a, best_b, min_error


# 4. Define a function to compute the average error for the fitted lines
def MeanSquareError(corner_groups, img_center, k1, k2 = None):
    error_sum = 0
    count = 0
    result = list()
    ud_groups = []
    for d_corners in corner_groups:

        ud_corners = apply_kappa(d_corners, img_center, k1, k2)
        ud_groups.append(ud_corners)
        a, b, error = RANSAC(ud_corners)
        result.append((a, b))
        error_sum += error
        count += ud_corners.shape[0]
    return error_sum / count, result

# 5. You need an iterative approach to brute force the kappa values
def default_iter(corner_data, img_center, k1_range, outer_iter_count, inner_iter_count, debug):
    min_k1, max_k1 = k1_range[0], k1_range[1]
    best_k1 = min_k1
    best_mse, best_linear = MeanSquareError(corner_data, img_center, best_k1)
    search_length = k1_range[1] - k1_range[0]
    while outer_iter_count > 0:
        is_improve = False
        for i in range(0, inner_iter_count + 1):
            next_k1 = min_k1 + (max_k1 - min_k1) * (i / inner_iter_count)
            next_mse, next_linear = MeanSquareError(corner_data, img_center, next_k1)
            if next_mse < best_mse:
                is_improve = True
                best_mse = next_mse
                best_linear = next_linear
                best_k1 = next_k1
                if debug:
                    print(f"[result improve] k1:{best_k1}, mse: {best_mse}")
        if is_improve:
            search_length *= 0.5
        else:
            search_length *= 0.25
        if search_length < sys.float_info.epsilon * inner_iter_count:
            break
        min_k1 = max(best_k1 - search_length / 2, k1_range[0])
        max_k1 = min(best_k1 + search_length / 2, k1_range[1])
        outer_iter_count -= 1
    return best_k1, best_mse, best_linear

def default_iter_k2(corner_data, img_center, k1, k2_range, outer_iter_count, inner_iter_count, previous_mse, previoous_linear, debug):
    min_k2, max_k2 = k2_range[0], k2_range[1]
    best_k2 = 0
    best_mse, best_linear = previous_mse, previoous_linear
    search_length = k2_range[1] - k2_range[0]
    while outer_iter_count > 0:
        is_improve = False
        for i in range(0, inner_iter_count + 1):
            next_k2 = min_k2 + (max_k2 - min_k2) * (i / inner_iter_count)
            next_mse, next_linear = MeanSquareError(corner_data, img_center, k1, next_k2)
            if next_mse < best_mse:
                is_improve = True
                best_mse = next_mse
                best_linear = next_linear
                best_k2 = next_k2
                if debug:
                    print(f"[result improve] k2:{best_k2}, mse: {best_mse}")
        if is_improve:
            search_length *= 0.5
        else:
            search_length *= 0.25
        min_k2 = max(best_k2 - search_length / 2, k2_range[0])
        max_k2 = min(best_k2 + search_length / 2, k2_range[1])
        outer_iter_count -= 1
    return best_k2, best_mse, best_linear


def undistort_img(img, corner_groups, debug=False):
    if debug:
        plot_image_with_lines(img, corner_groups)

    img_shape = img.shape
    img_center = np.array([img_shape[1], img_shape[0]]) / 2  # (width, height)
    best_k1, min_error, linear_result = default_iter(corner_groups, img_center, (0, 1e-6), 10, 50, debug)
    if debug:
        print(f"Find the best k1: {best_k1}, mean square error: {min_error}")

    best_k2, min_error, linear_result = default_iter_k2(corner_groups, img_center, best_k1, (-1e-13, 1e-13), 10, 50, min_error, linear_result, debug)
    if debug:
        print(f"Find the best k2 {best_k2}, mean square error: {min_error}")

    undistorted_points_group = np.array([apply_kappa(np.array(p), img_center, best_k1) for p in corner_groups])
    if debug:
        print("Generate undistorted image, This may takes some time")
    undistorted_img = A2_utilities.NearestNeighbourInterpolation(img, best_k1, best_k2)
    if debug:
        plot_image_with_point_group(undistorted_img, corner_groups, undistorted_points_group, linear_result, "Final Result")
    return {
        "img": undistorted_img,
        "points": undistorted_points_group.reshape(-1, 2),
        "k1": best_k1,
        "k2": best_k2
    }


def get_corner(img):
    pattern_size = (10, 7)
    world_space_points = []
    uv = []
    points3d2d = Get3d2dpoints(img, square_size=3.5, rows=pattern_size[0], columns=pattern_size[1], left_offset=3.4,  right_offset=3.4, debug=False)
    for item in points3d2d:
        world_space_points.append([item[0], item[1], item[2]])
        uv.append([item[3], item[4]])
    return np.array(uv), np.array(world_space_points)

def main():

    imgPath = "../Images/H3.JPG"
    img = utils.get_rgb_image(imgPath)
    uv, world_space_points = get_corner(np.copy(img))
    corners = uv.reshape(14, 10, 2)
    undistort_img(img, corners, True)







    # PLEASE DEFINE ALL YOUR FUNCTION IN UTILITIES AND IMPORT THEM HERE.


if __name__ == "__main__":
    main()
