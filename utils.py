import numpy as np
import cv2


def scaleImageTo0And255(image):
    min_v = np.min(image)
    max_v = np.max(image)
    if min_v != max_v:
        image = (image - min_v) / (max_v - min_v) * 255
    return image


def get_rgb_image(file_name):
    image = cv2.imread(file_name)
    if image.shape[0] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_gray_image(file_name, blur=True, scale_color=True):
    image = cv2.imread(file_name)
    if image.shape[0] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initial blurring of input images
    if blur:
        image = gaussian_filter(image, 5)

    # make sure greyscale image is stretched to range of 0 to 255
    if scale_color:
        image = scaleImageTo0And255(image)

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
