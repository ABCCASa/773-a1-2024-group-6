from matplotlib.patches import Circle, ConnectionPatch
from matplotlib import pyplot
import numpy as np


def plot_image_with_point_and_line(img1, img2,points1=None, points2=None, title=""):
    column_offset = img1.shape[1]
    width = img1.shape[1]+img2.shape[1]
    matchingImage = np.zeros((max(img1.shape[0], img2.shape[0]), width, 3))
    matchingImage[0:img1.shape[0], 0:img1.shape[1]] = img1
    matchingImage[0:img2.shape[0], column_offset:img2.shape[1]+column_offset] = img2
    matchingImage = matchingImage.astype(np.uint8)
    fig, ax = pyplot.subplots()
    ax.imshow(matchingImage)
    ax.set_title(title)
    point_offset = np.array([column_offset, 0])

    ys = []
    def can_place_line(ys, y_new):
        for y in ys:
            if abs(y-y_new)<100:
                return False
        ys.append(y_new)
        return True

    for i in range(len(points1)):
        p1 = points1[i]
        if can_place_line(ys, p1[1]):
            circle = Circle(p1, 10, color="r")
            ax.add_patch(circle)
            circle = Circle(points2[i]+point_offset, 10, color="b")
            ax.add_patch(circle)
            connection = ConnectionPatch((0, p1[1]), (width, p1[1]), "data", edgecolor='green')
            ax.add_patch(connection)

    pyplot.show()


def plot_two_img(img1, img2, with_line=False, title=""):
    column_offset = img1.shape[1]
    width = img1.shape[1]+img2.shape[1]
    matchingImage = np.zeros((max(img1.shape[0], img2.shape[0]), width, 3))
    matchingImage[0:img1.shape[0], 0:img1.shape[1]] = img1
    matchingImage[0:img2.shape[0], column_offset:img2.shape[1]+column_offset] = img2
    matchingImage = matchingImage.astype(np.uint8)
    fig, ax = pyplot.subplots()
    ax.imshow(matchingImage)
    if with_line:
        for i in range(0, img1.shape[0], 100):
            connection = ConnectionPatch((0,i), (width, i), "data", edgecolor='green')
            ax.add_patch(connection)




    ax.set_title(title)
    pyplot.show()
