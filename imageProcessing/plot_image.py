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
        points_copy = points.copy()
        left, right = None, None
        min_distance = None
        for i in range(len(points_copy) - 1):
            for j in range(i+1, len(points_copy)):
                temp_dis = point_distance2(points_copy[i], points_copy[j])
                if min_distance is None or temp_dis < min_distance:
                    left, right = points_copy[i], points_copy[j]
                    min_distance = temp_dis
        draw_line(left, right)
        points_copy.remove(left)
        points_copy.remove(right)

        while(len(points_copy) > 0):
            replace_left = True
            min_distance = None
            new_point = None
            for point in points_copy:
                temp_dis = point_distance2(left, point)
                if min_distance is None or temp_dis < min_distance:
                    replace_left = True
                    new_point = point
                    min_distance = temp_dis
                temp_dis = point_distance2(right, point)
                if min_distance is None or temp_dis < min_distance:
                    replace_left = False
                    new_point = point
                    min_distance = temp_dis
            points_copy.remove(new_point)
            if replace_left:
                draw_line(left, new_point)
                left = new_point
            else:
                draw_line(right, new_point)
                right = new_point

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
