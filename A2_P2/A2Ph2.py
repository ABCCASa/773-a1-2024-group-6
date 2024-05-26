from Get3d2dCoordinates import *
import pandas as pd
import utils
from img_polt import plot_image_with_point, plot_3d
import numpy as np
from A2_P1.A2Ph1 import undistort_img

# PLEASE DEFINE ALL YOUR FUNCTIONS IN UTILITIES AND IMPORT THEM HERE
def get_coords(file_name):
    uvs = []
    world_points = []
    df = pd.read_excel(file_name)
    for index, row in df.iterrows():
        uvs.append([row['u\''], row['v\'']])
        world_points.append([row["X"], row["Y"], row["Z"]])
    return np.array(uvs), np.array(world_points)


def uv_to_image_space_points(uv, img_center, delta, sx=1):
    camera_space_points = np.zeros_like(uv)
    camera_space_points[:, 0] = (uv[:, 0]-img_center[0])*delta/sx
    camera_space_points[:, 1] = (img_center[1]-uv[:, 1])*delta
    return camera_space_points


def get_r1_r2_tx_ty(abs_ty, sx,  world_space_points, image_space_points, L):
    for ty in [abs_ty, -abs_ty]:
        dis = np.sum((image_space_points**2), axis=1)
        idx = np.argmax(dis)
        far_img_point = image_space_points[idx]
        far_world_point = world_space_points[idx]
        r1 = L[:3]*ty
        r2 = L[4:]*ty
        tx = L[3] * ty / sx
        x, y = far_img_point[0], far_img_point[1]
        x_new = np.sum(r1*far_world_point)+tx
        y_new = np.sum(r2*far_world_point)+ty
        if ((y_new >= 0) == (y >= 0)) and ((x_new >= 0) == (x >= 0)):
            return r1, r2, tx, ty
    raise Exception("Can not find the sign of ty")


def get_l(image_space_points, world_space_points):
    m = []
    for isp, wsp in zip(image_space_points, world_space_points):
        m.append([
            isp[1] * wsp[0], isp[1] * wsp[1], isp[1] * wsp[2], isp[1], -isp[0] * wsp[0], -isp[0] * wsp[1], -isp[0] * wsp[2]
        ])
    m = np.array(m)
    xu = image_space_points[:, 0].reshape(-1, 1)
    m_inverse = np.linalg.pinv(m)
    L = np.dot(m_inverse, xu)
    return L.reshape(-1)


def get_f_tz(R, ty, image_space_point, world_space_point):
    x, y = [], []
    for isp, wsp in zip(image_space_point, world_space_point):
        x.append([np.sum(R[1]*wsp)+ty, -isp[1]])
        y.append([isp[1]*np.sum(wsp*R[2])])
    x, y = np.array(x), np.array(y)
    x_inverse = np.linalg.pinv(x)
    f_tz = np.dot(x_inverse, y).reshape(-1)
    return f_tz[0], f_tz[1]


def world_to_pixel(world_points, image_space_to_pixel_space, camera_space_to_image_space, world_to_camera_space_matrix):
    world_points = world_points.T
    world_points = np.concatenate((world_points, np.ones((1, world_points.shape[1]))), axis=0)
    uv = image_space_to_pixel_space@camera_space_to_image_space@world_to_camera_space_matrix@world_points
    uv = uv.T
    uv = uv[:, :2]/uv[:, 2, None]
    return uv


def uv_to_ray(uv, image_space_to_pixel_space, camera_space_to_image_space, world_to_camera_space_matrix, **kwargs):
    RT_inv = np.linalg.inv(world_to_camera_space_matrix)
    Owc = (RT_inv @ np.array([0, 0, 0, 1]).T)[:3].reshape(-1)
    camera_space_point = np.linalg.inv(image_space_to_pixel_space) @ np.concatenate([uv.T,
                          np.ones((1, uv.shape[0]))
                          ])
    uvs = np.concatenate([camera_space_point[:2],
                          camera_space_to_image_space[0, 0]*np.ones((1, uv.shape[0])),
                          np.ones((1, uv.shape[0]))
                          ])
    Mwi = np.dot(RT_inv, uvs).T[:, :3]
    return Owc, Mwi


def backproject(uv, image_space_to_pixel_space, camera_space_to_image_space, world_to_camera_space_matrix, axis):
    Owc, Mwi = uv_to_ray(uv, image_space_to_pixel_space, camera_space_to_image_space, world_to_camera_space_matrix)
    t = -Owc[axis]/(Mwi[:, axis]-Owc[axis])
    point_3d = (Mwi-Owc)*t[:, None] + Owc
    return point_3d



def get_corner_and_image(img):
    world_space_points = []
    uv = []
    pattern_size = (10, 7)
    points3d2d = Get3d2dpoints(img, square_size=3.5, rows=pattern_size[0], columns=pattern_size[1], left_offset=3.4,  right_offset=3.4, debug=False)
    for item in points3d2d:
        world_space_points.append([item[0], item[1], item[2]])
        uv.append([item[3], item[4]])
    return np.array(uv), np.array(world_space_points)


def Tsai_Calibration(img, pixel_size, need_undistort=True, debug=False):
    uv, world_space_points = get_corner_and_image(np.copy(img))
    if need_undistort:
        datas = undistort_img(img, uv.reshape(14, 10, 2), debug)
        img, uv = datas["img"], datas["points"]

    img_size = img.shape[:2]
    img_center = (img_size[1]/2, img_size[0]/2)
    image_space_points = uv_to_image_space_points(uv, img_center, pixel_size, 1)
    L = get_l(image_space_points, world_space_points)
    if debug:
        print(f"L={L}")
    abs_ty = 1 / np.sqrt(np.sum(L[4:7]**2))
    if debug:
        print(f"|ty|={ abs_ty}")
    sx = abs_ty*(np.sqrt(np.sum(L[0:3]**2)))
    if debug:
        print(f"sx={sx}")
    r1, r2, tx, ty = get_r1_r2_tx_ty(abs_ty, sx, world_space_points, image_space_points, L)
    if debug:
        print(f"tx={tx}, ty={ty}")
    r3 = np.cross(r1, r2)
    R = np.concatenate([r1, r2, r3]).reshape(3, 3)
    f, tz = get_f_tz(R, ty, image_space_points, world_space_points)
    if debug:
        print(f"f={f}, tz={tz}")
    t = np.array([tx, ty, tz])
    world_to_camera_space_matrix = np.zeros((4, 4))
    world_to_camera_space_matrix[:3, :3] = R
    world_to_camera_space_matrix[0, 3] = tx
    world_to_camera_space_matrix[1, 3] = ty
    world_to_camera_space_matrix[2, 3] = tz
    world_to_camera_space_matrix[3, 3] = 1

    if debug:
        print(f"world_to_camera_space_matrix:\n{world_to_camera_space_matrix}\n")
    camera_space_to_image_space = np.array(
        [[f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, 1, 0]]
    )

    if debug:
        print(f"camera_space_to_image_space:\n{camera_space_to_image_space}\n")
    image_space_to_pixel_space = np.array(
        [
            [sx/pixel_size, 0, img_center[0]],
            [0, -1/pixel_size, img_center[1]],
            [0, 0, 1]
        ]
    )
    if debug:
        print(f"image_space_to_pixel_space:\n{image_space_to_pixel_space}\n")

    world_space_to_pixel_space = image_space_to_pixel_space@camera_space_to_image_space@world_to_camera_space_matrix
    if debug:
        print(f"world_space_to_pixel_space\n{world_space_to_pixel_space}\n")

    return {
        "img": img,
        "uv": uv,
        'world_space_points': world_space_points,
        "image_space_to_pixel_space": image_space_to_pixel_space,
        "camera_space_to_image_space": camera_space_to_image_space,
        "world_to_camera_space_matrix": world_to_camera_space_matrix,
        "world_space_to_pixel_space": world_space_to_pixel_space,
        "f": f,
        "sx": sx
    }


# Test Error Part
def RMSE(point1, point2):
    return np.sqrt(np.mean((point1-point2)**2))


def test_2d_error(img, uv, world_space_points, image_space_to_pixel_space, camera_space_to_image_space, world_to_camera_space_matrix, **kwargs):
    uv_new = world_to_pixel(world_space_points, image_space_to_pixel_space, camera_space_to_image_space, world_to_camera_space_matrix)
    error = RMSE(uv, uv_new)
    print("The RMSE for 3d->2d is", error)
    plot_image_with_point(img, uv, uv_new, title=f"Projection from 3D coordinates to 2D pixels, MSE={error}")


def test_single_3d_error(uv, world_space_points, image_space_to_pixel_space, camera_space_to_image_space, world_to_camera_space_matrix, **kwargs):
    backproject_3d_point_left = backproject(uv[: uv.shape[0] // 2], image_space_to_pixel_space, camera_space_to_image_space,
                                            world_to_camera_space_matrix, 0)
    backproject_3d_point_right = backproject(uv[uv.shape[0] // 2:], image_space_to_pixel_space, camera_space_to_image_space,
                                             world_to_camera_space_matrix, 1)
    backproject_3d_point = np.concatenate((backproject_3d_point_left, backproject_3d_point_right)).reshape(-1, 3)
    error = RMSE(world_space_points, backproject_3d_point)
    print("The RMSE for 2d->3d is", error)
    plot_3d(world_space_points, backproject_3d_point, title=f"Projection from 2D coordinates to 3D pixels, MSE={error}")

def intersection(p1, p2, p3, p4):
    d21 = p2 - p1
    d34 = p3 - p4
    d13 = p1 - p3

    a = np.array([
            [d21.T@d21, d21.T@d34],
            [d21.T@d34, d34.T@d34]
        ])
    b = np.array([-d13.T@d21, -d13.T@d34]).T
    t = np.linalg.inv(a)@b

    return ((p1 + t[0]*(p2-p1)) + (p3+t[1]*(p4-p3)))/2


def stereo_calibration_error(data1, data2, world_point):
    Owc1, Mwi1 = uv_to_ray(**data1)
    Owc2, Mwi2 = uv_to_ray(**data2)

    inter_point = []
    for i in range(len(Mwi1)):
        inter_point.append(intersection(Owc1, Mwi1[i], Owc2, Mwi2[i]))

    inter_point = np.array(inter_point)
    error = RMSE(inter_point, world_point)
    plot_3d(world_point, inter_point, title=f"Stereo Calibration, MSE={error}")



def main():
    h3_l_img = utils.get_rgb_image("H3/Left/H3_Cube_Left.png")
    h3_l_data = Tsai_Calibration(img=h3_l_img, pixel_size=0.000155, need_undistort=True, debug=True)
    test_2d_error(**h3_l_data)
    test_single_3d_error(**h3_l_data)

    h3_r_img = utils.get_rgb_image("H3/Right/H3_Cube_Right.png")
    h3_r_data = Tsai_Calibration(img=h3_r_img, pixel_size=0.000155, need_undistort=True, debug=False)

    stereo_calibration_error(h3_l_data, h3_r_data, h3_l_data["world_space_points"])



if __name__ == "__main__":
    main()
