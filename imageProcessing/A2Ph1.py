import numpy as np
import cv2
from utilities import *
from CornerDetection.harris_corner_detection import harrisCornerDetection
import utils
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
import A2_utilities

def plot_image_with_point(image, points, title=""):
    fig, ax = pyplot.subplots()
    ax.set_title(title)
    ax.imshow(image, cmap='gray')
    for point in points:
        circle = Circle(point, 1, color='r')
        ax.add_patch(circle)
    pyplot.show()


def RANSAC(corners, img):
    img_shape = img.shape
    center = img_shape/2
    np_corners = []
    # convert (x, y) to (row, column)
    for c, r in corners:
        np_corners.append([r, c])
    distorted_corners = np.array(np_corners)





def apply_kappa(corners, center, k1):
    delta = corners-center
    distance = 1 + np.sum(delta**2, axis=1) * k1
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

    # 1. Import your own Harris corner detection to find all the corners in the chessboard. You will mark down if you use opencv "findChessboardCorners."
    img_gray = utils.get_gray_image(imgPath, blur=True, scale_color=True)
    corners = harrisCornerDetection(img_gray, count=500, window_size=5, alpha=0.06, threshold=500, nms_size=11)
    plot_image_with_point(img_gray, corners)

    # 2. You need to find a way to group the detected corner coordinates on horizontal lines in the chessboard image
    
    # 3. Define the RANSAC line-fitting algorithm to get the best slope and intercept of each horizontal line of points
    
    # 4. Define a function to compute the average error for the fitted lines
    
    # 5. You need an iterative approach to brute force the kappa values
    
    # 6. After estimating kappa, use "NearestNeighbourInterpolation" to undistort the original distorted image.



    # PLEASE DEFINE ALL YOUR FUNCTION IN UTILITIES AND IMPORT THEM HERE.
    
    
if __name__ == "__main__":
    main()
    
