import numpy as np
from matplotlib import pyplot
import utils
from matplotlib.patches import Circle



def harrisCornDetection(image, count, window_size=5, alpha=0.04, threshold=100, nms_size=3, debug=False):
    Ix = utils.sobel_filter(image, 'x')
    Iy = utils.sobel_filter(image, 'y')

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    # smoothing the squared derivatives
    smooth_Ixx = utils.gaussian_filter(Ixx, window_size)
    smooth_Iyy = utils.gaussian_filter(Iyy, window_size)
    smooth_Ixy = utils.gaussian_filter(Ixy, window_size)

    # C = det(H) - alpha*(trace(H)^2)
    detH = smooth_Ixx*smooth_Iyy-smooth_Ixy**2
    traceH = smooth_Ixx+smooth_Iyy
    C = detH - alpha * (traceH ** 2)
    C_threshold = np.copy(C)
    C_threshold[C_threshold < threshold] = 0

    # non-maxima suppression
    corners = []
    center_mask = np.ones((nms_size, nms_size), dtype=bool)
    nms_offset = nms_size//2
    center_mask[nms_offset, nms_offset] = False  # The center point needs to be the unique maximum

    for row in range(nms_offset, C.shape[0]-nms_offset):
        for column in range(nms_offset, C.shape[1]-nms_offset):
            center_val = C[row, column]
            if center_val >= threshold:
                nms_window = C[row-nms_offset:row+nms_offset+1, column-nms_offset:column+nms_offset+1]
                if np.max(nms_window[center_mask]) < center_val:
                    corners.append((C[row, column], (column, row)))

    # sort and get top x corners
    corners.sort(key=lambda x: x[0], reverse=True)
    top_x_corners = [x[1] for x in corners[:min(len(corners), count)]]

    if debug:
        fig, axs = pyplot.subplots(3, 4)
        for i in range(3):
            for j in range(4):
                axs[i][j].axis('off')
        axs[0][0].set_title('Ix')
        axs[0][0].imshow(Ix, cmap='gray')
        axs[0][1].set_title('Iy')
        axs[0][1].imshow(Iy, cmap='gray')
        axs[0][2].set_title('Ixx')
        axs[0][2].imshow(Ixx, cmap='gray')
        axs[0][3].set_title('Iyy')
        axs[0][3].imshow(Iyy, cmap='gray')
        axs[1][0].set_title('Ixy')
        axs[1][0].imshow(Ixy, cmap='gray')
        axs[1][1].set_title('g_Ixx')
        axs[1][1].imshow(smooth_Ixx, cmap='gray')
        axs[1][2].set_title('g_Iyy')
        axs[1][2].imshow(smooth_Iyy, cmap='gray')
        axs[1][3].set_title('g_Ixy')
        axs[1][3].imshow(smooth_Ixy, cmap='gray')
        axs[2][0].set_title('C')
        axs[2][0].imshow(C, cmap='gray')
        axs[2][1].set_title('C_threshold')
        axs[2][1].imshow(C_threshold, cmap='gray')
        axs[2][2].set_title('C_polt')
        axs[2][2].imshow(C_threshold, cmap='gray')
        axs[2][3].set_title('image_polt')
        axs[2][3].imshow(image, cmap='gray')
        for point in top_x_corners:
            axs[2][2].add_patch(Circle(point, 0.5, color='r'))
            axs[2][3].add_patch(Circle(point, 0.5, color='r'))
        pyplot.show()
    return top_x_corners