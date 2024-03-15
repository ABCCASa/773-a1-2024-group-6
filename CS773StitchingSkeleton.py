import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch

from timeit import default_timer as timer

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth
import imageProcessing.convolve2D as IPConv2D

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


def convolution(image, kernel):
    result = np.ones(image.shape)
    kernel_size = kernel.shape[0]
    image = np.pad(image, kernel_size // 2, mode="edge")
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.sum(kernel*image[i:i+kernel_size, j:j+kernel_size])
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
    return convolution(image, kernel)/8

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size,size))
    offset = size//2
    for x in range(size):
        for y in range(size):
            kernel[x, y] = (1/(2*np.pi*sigma**2)) * np.exp(-((x-offset)**2 + (y-offset)**2) / (2*sigma**2))
    kernel = kernel/np.sum(kernel)
    return kernel


def gaussian_filter(image, size, sigma=1):
    kernel = gaussian_kernel(size,sigma)
    return convolution(image, kernel)


def get_image_corner(filename_image, alpha=0.04):
    image_width, image_height, px_array_original = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_image)
    image = np.array(px_array_original)

    # smooth image
    image = gaussian_filter(image, 3)

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    min_v = np.min(image)
    max_v = np.max(image)
    if min_v != max_v:
        image = (image - min_v) / (max_v - min_v) * 255
    Ix = sobel_filter(image, 'x')
    Iy = sobel_filter(image, 'y')

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    Ixx = gaussian_filter(Ixx, 5)
    Iyy = gaussian_filter(Iyy, 5)
    Ixy = gaussian_filter(Ixy, 5)

    # C = det(H) - alpha*(trace(H)^2)
    detH = Ixx*Iyy-Ixy**2
    traceH = Ixx+Iyy
    C = detH - alpha * (traceH ** 2)

    # non-maxima suppression
    corners = []
    centerMask = np.array([[True, True, True],
                           [True, False, True],
                           [True, False, True]
                           ])
    for row in range(1, C.shape[0]-1):
        for column in range(1, C.shape[1]-1):
            if C[row, column] > 0 and np.max(C[row-1:row+2, column-1:column+2][centerMask]) < C[row, column]:
                corners.append((C[row, column], (row, column)))

    # sort and get top 1000 corners
    corners.sort(key=lambda x: x[0], reverse=True)
    top_1000_corners = []

    for i in range(min(len(corners), 1000)):
        value, point = corners[i]
        top_1000_corners.append(point)

    # Plot image - The coordinates of a point is (row, column), please convert to (x, y) when using polt.
    fig, ax = pyplot.subplots()
    ax.imshow(image, cmap='gray')
    for point in top_1000_corners:
        circle = Circle((point[1], point[0]), 1, color='r')
        ax.add_patch(circle)
    pyplot.show()

    return top_1000_corners




# This is our code skeleton that performs the stitching
def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    # get corner of image, the coordinates of a corner is (row, column), remember convert to (x, y) when using polt.
    left_corner = get_image_corner(filename_left_image)
    right_corner = get_image_corner(filename_right_image)
"""
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

