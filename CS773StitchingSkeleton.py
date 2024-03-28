import time

import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
from timeit import default_timer as timer
import imageIO.readwrite as IORW
import imageProcessing.utilities as IPUtils
import extension as ext

# this is a helper function that puts together an RGB image for display in matplotlib, given
# three color channels for r, g, and b, respectively
def prepareRGBImageFromIndividualArrays(r_pixel_array, g_pixel_array, b_pixel_array, image_width, image_height):
    rgbImage = []
    for y in range(image_height):
        row = []
        for x in range(image_width):
            triple = []
            triple.append(r_pixel_array[y][x])
            triple.append(g_pixel_array[y][x])
            triple.append(b_pixel_array[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):
    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width * 2, image_height)
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]
    return matchingImage


def plot_image_with_point(image, points):
    fig, ax = pyplot.subplots()
    ax.imshow(image, cmap='gray')
    for point in points:
        circle = Circle(point, 0.5, color='r')
        ax.add_patch(circle)
    pyplot.show()


def scaleImageTo0And255(image):
    min_v = np.min(image)
    max_v = np.max(image)
    if min_v != max_v:
        image = (image - min_v) / (max_v - min_v) * 255
    return image


# Faster convolution calculations
def convolution_2d(image, kernel, pad_mode="edge"):
    img_shape = image.shape
    result = np.zeros(img_shape)
    kernel_size = kernel.shape[0]
    image = np.pad(image, kernel_size//2, mode=pad_mode)
    for r in range(kernel_size):
        for c in range(kernel_size):
            result += image[r:img_shape[0] + r, c:img_shape[1] + c]*kernel[r, c]
    return result


def sobel_filter(image, axis):
    if axis == 'x':
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    elif axis == 'y':
        kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
    return convolution_2d(image, kernel)/8


def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    offset = size//2
    for x in range(size):
        for y in range(size):
            kernel[x, y] = (1/(2*np.pi*sigma**2)) * np.exp(-((x-offset)**2 + (y-offset)**2) / (2*sigma**2))
    kernel = kernel/np.sum(kernel)
    return kernel


def gaussian_filter(image, size, sigma=1):
    kernel = gaussian_kernel(size, sigma)
    return convolution_2d(image, kernel)


def get_image(file_name, blur=True, scale_color=True):
    _, _, px_array_original = IORW.readRGBImageAndConvertToGreyscalePixelArray(file_name)

    # convert image to np.array
    image = np.array(px_array_original)

    # Initial blurring of input images
    if blur:
        image = gaussian_filter(image, 3)

    # make sure greyscale image is stretched to range of 0 to 255
    if scale_color:
        image = scaleImageTo0And255(image)

    return image


def image_corner_detection(image, count, window_size=5, alpha=0.04, threshold=100, nms_size=3, debug=False):
    Ix = sobel_filter(image, 'x')
    Iy = sobel_filter(image, 'y')

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    # smoothing the squared derivatives
    smooth_Ixx = gaussian_filter(Ixx, window_size)
    smooth_Iyy = gaussian_filter(Iyy, window_size)
    smooth_Ixy = gaussian_filter(Ixy, window_size)

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


# This is our code skeleton that performs the stitching
def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    left_image = get_image(filename_left_image)
    right_image = get_image(filename_right_image)


    left_corner = image_corner_detection(left_image, count=1000,  alpha=0.04, threshold=100, nms_size=3, debug=True)
    plot_image_with_point(left_image, left_corner)

    right_corner = image_corner_detection(right_image, count=1000,alpha=0.04, threshold=100, nms_size=3, debug=True)
    plot_image_with_point(right_image, right_corner)

    left_corner_fast_extension = ext.fast_corner_detection(left_image)
    plot_image_with_point(left_image, left_corner_fast_extension)

    right_corner_fast_extension = ext.fast_corner_detection(right_image)
    plot_image_with_point(right_image, right_corner_fast_extension)

    """
    
    left_corner = image_corner_detection(left_image, count=1000, alpha=0.04, threshold=100, nms_size=3, debug=True)
    plot_image_with_point(left_image, left_corner)

    right_corner = image_corner_detection(right_image, count=1000, alpha=0.04, threshold=100, nms_size=3, debug=True)
    plot_image_with_point(right_image, right_corner)
    
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    (image_width, image_height, px_array_left_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_left_image)
    (image_width, image_height, px_array_right_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_right_image)

    start = timer()
    px_array_left = IPSmooth.computeGaussianAveraging3x3(px_array_left_original, image_width, image_height)
    px_array_right = IPSmooth.computeGaussianAveraging3x3(px_array_right_original, image_width, image_height)
    end = timer()
    print("elapsed time image smoothing: ", end - start)

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_array_left = IPPixelOps.scaleTo0And255AndQuantize(px_array_left, image_width, image_height)
    px_array_right = IPPixelOps.scaleTo0And255AndQuantize(px_array_right, image_width, image_height)

    # some simplevisualizations
    fig1, axs1 = pyplot.subplots(1, 2)

    axs1[0].set_title('Harris response left overlaid on orig image')
    axs1[1].set_title('Harris response right overlaid on orig image')
    axs1[0].imshow(px_array_left, cmap='gray')
    axs1[1].imshow(px_array_right, cmap='gray')

    # plot a red point in the center of each image
    circle = Circle((image_width/2, image_height/2), 3.5, color='r')
    axs1[0].add_patch(circle)

    circle = Circle((image_width/2, image_height/2), 3.5, color='r')
    axs1[1].add_patch(circle)

    pyplot.show()

    # a combined image including a red matching line as a connection patch artist (from matplotlib)

    matchingImage = prepareMatchingImage(px_array_left, px_array_right, image_width, image_height)

    pyplot.imshow(matchingImage, cmap='gray')
    ax = pyplot.gca()
    ax.set_title("Matching image")

    pointA = (image_width/2, image_height/2)
    pointB = (3*image_width/2, image_height/2)
    connection = ConnectionPatch(pointA, pointB, "data", edgecolor='r', linewidth=1)
    ax.add_artist(connection)

    pyplot.show()
    """


if __name__ == "__main__":
    main()

