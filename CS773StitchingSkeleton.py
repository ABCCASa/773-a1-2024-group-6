import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
from timeit import default_timer as timer
import imageProcessing.utilities as IPUtils
from CornerDetection.fast_corner_detection import fastCornerDetection
from CornerDetection.harris_corner_detection import harrisCornerDetection
from FeatureMatch.ncc_match import NCCMatch
import utils
from ImageStitching.fitting import projective_transformation_fitting
from ImageStitching.warping import warping


# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
def PoltMatchingImage(left_image, right_image, matched_points, title=""):
    column_offset = left_image.shape[1]
    matchingImage = np.zeros((max(left_image.shape[0], right_image.shape[0]), left_image.shape[1]+right_image.shape[1]))
    matchingImage[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    matchingImage[0:right_image.shape[0], column_offset:right_image.shape[1]+column_offset] = right_image
    fig, ax = pyplot.subplots()
    ax.set_title(title)
    ax.imshow(matchingImage, cmap='gray')
    for left_point, right_point in matched_points:
        right_point = (right_point[0]+column_offset, right_point[1])
        connection = ConnectionPatch(left_point, right_point, "data", edgecolor='r', linewidth=1)
        ax.add_patch(connection)
    pyplot.show()


def plot_image_with_point(image, points, title=""):
    fig, ax = pyplot.subplots()
    ax.set_title(title)
    ax.imshow(image, cmap='gray')
    for point in points:
        circle = Circle(point, 0.5, color='r')
        ax.add_patch(circle)
    pyplot.show()


def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    left_image = utils.get_gray_image(filename_left_image)
    right_image = utils.get_gray_image(filename_right_image)

    """ ***********************************  Corner Detection  *************************************************** """
    left_corner = harrisCornerDetection(left_image, count=1000,  alpha=0.04, threshold=100, nms_size=3, debug=False)
    # plot_image_with_point(left_image, left_corner, "Harris Corner Detection: Left Image")

    right_corner = harrisCornerDetection(right_image, count=900, alpha=0.04, threshold=100, nms_size=3, debug=False)
    # plot_image_with_point(right_image, right_corner, "Harris Corner Detection: Right Image")

    """
    left_corner_fast_extension = fastCornerDetection(left_image)
    plot_image_with_point(left_image, left_corner_fast_extension, "FAST Corner Detection: Left Image")

    right_corner_fast_extension = fastCornerDetection(right_image)
    plot_image_with_point(right_image, right_corner_fast_extension, "FAST Corner Detection: Right Image")
    """

    """ ***********************************  Feature Match  *************************************************** """
    matched_Point = NCCMatch(left_image, left_corner, right_image, right_corner, window_size=15, ratio_threshold=0.90)
    PoltMatchingImage(left_image, right_image, matched_Point)




if __name__ == "__main__":
    main()

