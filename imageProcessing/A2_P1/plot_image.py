from matplotlib.patches import Circle, ConnectionPatch
from matplotlib import pyplot



def plot_image_with_point(image, points, title=""):
    fig, ax = pyplot.subplots()
    ax.set_title(title)
    ax.imshow(image, cmap='gray')
    for point in points:
        circle = Circle(point, 2, color='r')
        ax.add_patch(circle)
    pyplot.show()


def point_distance2(p1, p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2


def plot_image_with_lines(image, points_group, title=""):
    fig, ax = pyplot.subplots()
    ax.set_title(title)
    ax.imshow(image, cmap='gray')

    def draw_line(p1, p2):
        connection = ConnectionPatch(p1, p2, "data", edgecolor='b')
        ax.add_patch(connection)

    for points in points_group:
        for i in range(len(points)-1):
            draw_line(points[i], points[i+1])

        for point in points:
            circle = Circle(point, 2, color='r')
            ax.add_patch(circle)
    pyplot.show()






def plot_image_with_point_group(image, points_group1=None, points_group2=None, linears=None, title=""):
    width = image.shape[1]
    height = image.shape[0]
    fig, ax = pyplot.subplots()
    ax.set_title(title)

    if image is not None:
        ax.imshow(image, cmap='gray')

    if linears is not None:
        for a, b in linears:
            min_x = max(min(width, -b / a), 0)
            max_x = max(min(width, (height - b) / a), 0)
            connection = ConnectionPatch((min_x, a * min_x + b), (max_x, a * max_x + b), "data", edgecolor='green')
            ax.add_patch(connection)

    if points_group1 is not None:
        for points in points_group1:
            for point in points:
                circle = Circle(point, 3, color='r')
                ax.add_patch(circle)

    if points_group2 is not None:
        for points in points_group2:
            for point in points:
                circle = Circle(point, 2, color='b')
                ax.add_patch(circle)
    pyplot.show()
