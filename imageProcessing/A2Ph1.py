import numpy as np
import random
from utilities import *
from CornerDetection.harris_corner_detection import harrisCornerDetection
import utils
import A2_utilities
import math
import line_select_window


from plot_image import plot_image_with_lines, plot_image_with_point, plot_image_with_point_group


def apply_kappa(corners, center, k1, k2=None):
    delta = corners - center
    distance = 1 + np.sum(delta ** 2, axis=1) * k1
    if k2 is not None:
        distance += np.sum(delta ** 4, axis=1) * k2
    return delta * distance[:, None] + center



def main():
    # PHASE-2-ASSIGNMENT-1: distortion removal
    # Given a chessboard image, you need to undistort the image by the assumption that straight lines in the real world are expected to be straight in the undistorted image.
    # The first step is to estimate kappa (k1) using an iterative algorithm based on RANSAC line fitting.
    # Refer to the lecture and tutorial notes for more details.
    # Once k1 is estimated you can use "NearestNeighbourInterpolation" function to undistort the original image
    # The provided function "NearestNeighbourInterpolation" needs the distorted image and the estimated kappa as inputs, and it returns the undistorted image.

    imgPath = "../Images/H3.JPG"
    img_rgb = utils.get_rgb_image(imgPath)
    img_gray = utils.get_gray_image(imgPath, blur=True, scale_color=True)
    # 1. Import your own Harris corner detection to find all the corners in the chessboard. You will mark down if you use opencv "findChessboardCorners."
    board_left = (555, 267, 1650, 1750)
    board_right = (1650, 267, 2750, 1850)

    def point_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_board_corner(row_image, area, count):
        board_img = row_image[area[1]: area[3], area[0]: area[2]]
        board_corners = harrisCornerDetection(board_img, count=count, window_size=7, alpha=0.06, threshold=500, nms_size=21)
        image_corner = [(x+area[0], y+area[1]) for x, y in board_corners]
        return image_corner

    left_board_corners = get_board_corner(img_gray, board_left, 101)
    right_board_corners = get_board_corner(img_gray, board_right, 106)
    plot_image_with_point(img_gray, left_board_corners+right_board_corners, "Row Chessboard Corners")

    # 2. You need to find a way to group the detected corner coordinates on horizontal lines in the chessboard image
    def remove_too_close_points(corners, threshold=40):
        is_far = [True]*len(corners)
        for i in range(len(corners)):
            for j in range(i+1, len(corners)):
                if point_distance(corners[i], corners[j]) < threshold:
                    is_far[i], is_far[j] = False, False
        output = []
        for i in range(len(corners)):
            if is_far[i]:
                output.append(corners[i])
        return output

    def make_line(corners, max_distance = 180,max_slope=0.8, line_count=5):
        sort_corners = sorted(corners)
        pairs = []
        for i in range(len(sort_corners)-1):
            p1 = sort_corners[i]
            min_d = None
            min_p2 = None
            for j in range(i+1, len(sort_corners)):
                p2 = sort_corners[j]
                d = point_distance(p1, p2)
                if p2[0] - p1[0] != 0:
                    slope = abs((p2[1] - p1[1]) / (p2[0] - p1[0]))
                    if d < max_distance and slope < max_slope:
                        if min_d is None or d < min_d:
                            min_d = d
                            min_p2 = p2
            if min_p2 is not None:
                pairs.append({p1, min_p2})
        lines = []
        for pair in pairs:
            is_add = False
            for line in lines:
                if len(pair.intersection(line)) > 0:
                    line.update(pair)
                    is_add = True
            if not is_add:
                lines.append(pair)
        return [list(line) for line in lines if len(line) >= line_count]

    """Remove the points that are too close to each other"""
    left_board_corners = remove_too_close_points(left_board_corners)
    right_board_corners = remove_too_close_points(right_board_corners)

    """group the corners as lines"""
    left_corner_groups = make_line(left_board_corners)
    right_corner_groups = make_line(right_board_corners)
    corner_groups = left_corner_groups + right_corner_groups

    """plot data"""
    plot_image_with_lines(img_gray, corner_groups, "Grouped Corners")
    """
    row_corners = left_board_corners + right_board_corners
    corner_groups = line_select_window.select_line(img_gray, row_corners, imgPath)
    """

    """convert tuple to array"""
    for i in range(len(corner_groups)):
        corner_groups[i] = np.array(corner_groups[i])

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

    def RANSAC(corners, iters=None):
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
            a, b, error = RANSAC(ud_corners, 100)
            result.append((a, b))
            error_sum += error
            count += ud_corners.shape[0]
        # real_time_debug(fig, ax, img_gray, corner_groups, ud_groups, result)

        return error_sum / count, result

    # 5. You need an iterative approach to brute force the kappa values

    def simulated_annealing(corner_data, img_center, k1_range, step, T0=100, Tf=1e-4, alpha=0.98):
        current_k1 = random.uniform(k1_range[0], k1_range[1])
        current_mse, current_linear = MeanSquareError(corner_data, img_center, current_k1)
        best_k1, best_mse, best_linear = current_k1, current_mse, current_linear

        T = T0
        while T > Tf:
            next_k1 = current_k1 + random.uniform(-1, 1) * step * T
            next_k1 = max(k1_range[0], min(k1_range[1], next_k1))
            next_mse, next_linear = MeanSquareError(corner_data, img_center, next_k1)
            E = current_mse - next_mse
            if E >= 0:
                current_k1, current_mse, current_linear = next_k1, next_mse, next_linear
                best_k1, best_mse, best_linear = current_k1, current_mse, current_linear
                print(f"[result improve] k1:{best_k1}, mse: {best_mse}")
            elif random.uniform(0, 1) < math.pow(math.e, E / T):
                current_k1, current_mse, current_linear = next_k1, next_mse, next_linear
            T *= alpha
        return best_k1, best_mse, best_linear

    def hill_climbing(corner_data, img_center, k1_range, step, alpha, iter_count):
        best_k1 = random.uniform(k1_range[0], k1_range[1])
        best_mse, best_linear = MeanSquareError(corner_data, img_center, best_k1)

        for i in range(iter_count):
            next_k1 = best_k1 + step * random.uniform(-1, 1)
            next_k1 = max(k1_range[0], min(k1_range[1], next_k1))
            next_mse, next_linear = MeanSquareError(corner_data, img_center, next_k1)
            if best_mse >= next_mse:
                best_k1, best_mse, best_linear = next_k1, next_mse, next_linear
                print(f"[result improve] k1:{best_k1}, mse: {best_mse}")
            step *= alpha
        return best_k1, best_mse, best_linear

    def default_iter(corner_data, img_center, k1_range, outer_iter_count, inner_iter_count):
        min_k1, max_k1 = k1_range[0], k1_range[1]
        best_k1 = min_k1
        best_mse, best_linear = MeanSquareError(corner_groups, img_center, best_k1)
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

    def default_iter_k2(corner_data, img_center, k1, k2_range, outer_iter_count, inner_iter_count, previous_mse, previoous_linear):
        min_k2, max_k2 = k2_range[0], k2_range[1]
        best_k2 = 0
        best_mse, best_linear = previous_mse, previoous_linear
        print(best_mse)
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
                    print(f"[result improve] k2:{best_k2}, mse: {best_mse}")
            if is_improve:
                search_length *= 0.5
            else:
                search_length *= 0.25
            min_k2 = max(best_k2 - search_length / 2, k2_range[0])
            max_k2 = min(best_k2 + search_length / 2, k2_range[1])
            outer_iter_count -= 1
        return best_k2, best_mse, best_linear

    img_shape = img_gray.shape
    img_center = np.array([img_shape[1], img_shape[0]]) / 2  # (width, height)
    best_k1, min_error, linear_result = default_iter(corner_groups, img_center, (0, 1e-6), 20, 200)
    print(f"Find the best k1: {best_k1}, mean square error: {min_error}")

    best_k2, min_error, linear_result = default_iter_k2(corner_groups, img_center, best_k1, (-1e-13, 1e-13), 20, 200, min_error, linear_result)
    print(f"Find the best k2 {best_k2}, mean square error: {min_error}")

    # best_k1, min_error, linear_result = simulated_annealing(corner_groups, img_center, (0, 1e-6), step=1e-7)
    # print(f"Find the best kappa {best_k1}, mean square error: {min_error}")

    # best_k1, min_error, linear_result = hill_climbing(corner_groups, img_center, (0, 1e-6), 1e-7, 0.99, 3000)
    # print(f"Find the best kappa {best_k1}, mean square error: {min_error}")

    # 6. After estimating kappa, use "NearestNeighbourInterpolation" to undistort the original distorted image.

    undistorted_points_group = [apply_kappa(np.array(p), img_center, best_k1) for p in corner_groups]
    undistorted_img = A2_utilities.NearestNeighbourInterpolation(img_rgb, best_k1, best_k2)
    plot_image_with_point_group(undistorted_img, corner_groups, undistorted_points_group, linear_result, "Final Result")

    # PLEASE DEFINE ALL YOUR FUNCTION IN UTILITIES AND IMPORT THEM HERE.


if __name__ == "__main__":
    main()
