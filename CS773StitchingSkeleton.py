import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
from timeit import default_timer as timer
import imageProcessing.utilities as IPUtils
from CornerDetection.fast_corner_detection import fastCornerDetection
from CornerDetection.harris_corner_detection import harrisCornerDetection
from FeatureMatch.ncc_match import NCCMatch
from FeatureMatch.ssda_match import SSDAMatch
from ImageStitching.fitting import projective_transformation_fitting
from ImageStitching.warping import warping
import utils

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
def PoltMatchingImage(left_image, right_image, matched_points,title=""):
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
    matched_Point = NCCMatch(left_image, left_corner, right_image, right_corner)
    PoltMatchingImage(left_image, right_image, matched_Point,"NCC Match")

    """
    matched_Point = SSDAMatch(left_image, left_corner, right_image, right_corner)
    PoltMatchingImage(left_image, right_image, matched_Point,"SSDA Match")
    """


    """ ***********************************  Image Stitching  *************************************************** """
    left_image = utils.get_rgb_image(filename_left_image)
    right_image = utils.get_rgb_image(filename_right_image)

   
    H, inliers_count = projective_transformation_fitting(matched_Point, distance_threshold=2, iter_count=100000)
    print(f"get {inliers_count} inliers, which is {int(100*inliers_count/len(matched_Point))}% of all points")


    final_image = warping(left_image, [(right_image, H)])
    pyplot.imshow(final_image)
    pyplot.show()


if __name__ == "__main__":
    main()

