# We applied FAST corner detection as our extension
# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e3d7f693bf2d3510a0557cda52c7547820fbef97
import numpy as np


# Returns the coordinates of a continuous circle
def neighbor_circle():
    return[(0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)]


# return the score of the
def get_score_and_count(img, t, n):
    max_count = None
    current_count = np.ones(img.shape)
    last_state = None  # -1=darker 0=similar 1=brighter
    score = np.zeros(img.shape)
    img_pad = np.pad(img, 3, mode="edge")
    circle_coord = neighbor_circle()

    for i in range(len(circle_coord)+n-1):
        x, y = circle_coord[i % len(circle_coord)]
        shift_img = img_pad[3+x: img.shape[0]+x+3, 3+y:img.shape[1]+y+3]
        delta_img = shift_img-img
        current_state = np.zeros(img.shape, dtype=np.int32)
        current_state[delta_img > t] = 1
        current_state[delta_img < -t] = -1

        if i < len(circle_coord):
            score += np.abs(delta_img)

        if last_state is not None:
            # if state keep same, count +1
            current_count[last_state == current_state] += 1
            # if state is change, set value to 1
            current_count[last_state != current_state] = 1
            # if is similar set count to 0
            current_count[current_state == 0] = 0
            # keep the max count
            max_count = np.maximum(current_count, max_count)
        else:
            # init data if is the first value
            current_count[current_state != 0] = 1
            max_count = current_count

        last_state = current_state

    return score, max_count


# FAST Corner Detection
def fastCornerDetection(image, t=30, n=9, nms_size=5, top_x=1000):
    score, count = get_score_and_count(image, t, n)
    isCorner = count >= n
    center_mask = np.ones((nms_size, nms_size), dtype=bool)
    nms_offset = nms_size // 2
    center_mask[nms_offset, nms_offset] = False
    corners = []
    for row in range(nms_offset, image.shape[0]-nms_offset):
        for column in range(nms_offset, image.shape[1]-nms_offset):
            if isCorner[row, column]:
                center_val = score[row, column]
                nms_window = score[row-nms_offset:row+nms_offset+1, column-nms_offset:column+nms_offset+1]

                if np.max(nms_window[center_mask]) < center_val:
                    corners.append((score[row, column], (column, row)))

    # sort and get top x corners
    corners.sort(key=lambda x: x[0], reverse=True)
    top_x_corners = [x[1] for x in corners[:min(len(corners), top_x)]]
    return top_x_corners





