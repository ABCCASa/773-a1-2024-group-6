from matplotlib.patches import Circle, ConnectionPatch
from matplotlib import pyplot

def plot_image_with_point(image=None, points1=None, points2=None, title=""):
    fig, ax = pyplot.subplots()
    ax.set_title(title)
    if image is not None:
        ax.imshow(image, cmap='gray')
    if points1 is not None:
        for point in points1:
            circle = Circle(point, 10, color=(1, 0, 0, 0.5))
            ax.add_patch(circle)
    if points2 is not None:
        for point in points2:
            circle = Circle(point, 10, color=(0, 0, 1, 0.5))
            ax.add_patch(circle)
    pyplot.show()

def plot_3d(data0, data1=None, show_center=False, title = ""):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data0[:, 0], data0[:, 1], data0[:, 2], edgecolors="blue")
    if data1 is not None:
        ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], edgecolors="red")
    if show_center:
        ax.scatter(0, 0, 0, edgecolors="green")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    pyplot.show()