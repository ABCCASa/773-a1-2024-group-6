import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
import imageProcessing.utilities as IPUtils
from CornerDetection.fast_corner_detection import fastCornerDetection
from CornerDetection.harris_corner_detection import harrisCornerDetection
from FeatureMatch.ncc_match import NCCMatch
import utils
from ImageStitching.fitting import projective_transformation_fitting
from ImageStitching.warping import warping
import time


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


def plot_images_with_points(image_1, points_1, image_2, points_2, title_1="", title_2=""):
    fig, axs = pyplot.subplots(1, 2)
    plot_image_with_point(axs[0], image_1, points_1, title=title_1)
    plot_image_with_point(axs[1], image_2, points_2, title=title_2)
    pyplot.show()


def plot_image_with_point(ax, image, points, title):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    for point in points:
        circle = Circle(point, 0.5, color='r')
        ax.add_patch(circle)



def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    left_image = utils.get_gray_image(filename_left_image)
    right_image = utils.get_gray_image(filename_right_image)

    """ ***********************************  Corner Detection  *************************************************** """
    current_time = time.time()
    left_corner = harrisCornerDetection(left_image, count=1000,  alpha=0.04, threshold=100, nms_size=3, debug=False)
    print(f"Get {len(left_corner)} corners from left image")
    right_corner = harrisCornerDetection(right_image, count=1000, alpha=0.04, threshold=100, nms_size=3, debug=False)
    print(f"Get {len(right_corner)} corners from right image")
    print(f"Corner detection complete, use time {round(time.time()-current_time,2)}s\n")
    plot_images_with_points(left_image, left_corner, right_image,
                            right_corner, title_1="Left Image Corners", title_2="Right Image Corners")

    """
    left_corner_fast_extension = fastCornerDetection(left_image)
    plot_image_with_point(left_image, left_corner_fast_extension, "FAST Corner Detection: Left Image")

    right_corner_fast_extension = fastCornerDetection(right_image)
    plot_image_with_point(right_image, right_corner_fast_extension, "FAST Corner Detection: Right Image")
    """

    """ ***********************************  Feature Match  *************************************************** """
    current_time = time.time()
    matched_Point = NCCMatch(left_image, left_corner, right_image, right_corner,
                             window_size=15, ratio_threshold=0.9, two_dir_match=False)
    print(f"get {len(matched_Point)} matched points")
    print(f"Feature match complete, use time {round(time.time() - current_time, 2)}s\n")
    PoltMatchingImage(left_image, right_image, matched_Point, "Match Feature of two image")

   


if __name__ == "__main__":
    main()

