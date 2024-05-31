from A2_P2.A2Ph2 import Tsai_Calibration
import img_polt
import cv2
import numpy as np
import matplotlib.pyplot as pyplot
def img_warp(H, img, uv=None):
    warped_image = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    if uv is not None:
        uv = np.concatenate(
            [
                uv.T,
                np.ones((1, uv.shape[0]))
            ]
        )
        uv = H@uv
        uv = uv.T
        uv = uv[:, :2]/uv[:, 2, None]
    return warped_image, uv


def rectification(img1, img2,  pixel_size, need_undistort, generate_image, debug):
    data1 = Tsai_Calibration(img1,  pixel_size, need_undistort=need_undistort, generate_img=generate_image, debug=debug)
    data2 = Tsai_Calibration(img2,  pixel_size, need_undistort=need_undistort, generate_img=generate_image, debug=debug)
    if need_undistort and generate_image:
        img1, img2 = data1['img'], data2['img']

    M1 = data1["world_to_camera_space_matrix"]
    M2 = data2["world_to_camera_space_matrix"]
    K1 = data1["image_space_to_pixel_space"]@data1["camera_space_to_image_space"]
    K2 = data2["image_space_to_pixel_space"]@data2["camera_space_to_image_space"]
    H1, H2, b = _rectification(M1, M2, K1, K2)

    if generate_image:
        img1, uv1 = img_warp(H1, img1, data1['uv'])
        img2, uv2 = img_warp(H2, img2, data2['uv'])
        data1["img"] = img1
        data2["img"] = img2
        data1["uv"] = uv1
        data2["uv"] = uv2
        if debug:
            img_polt.plot_image_with_point_and_line(data1["img"], data2["img"], data1["uv"], data2["uv"])

    data1["H"] = H1
    data2["H"] = H2
    data1["b"] = b
    data2["b"] = b
    return data1, data2


def _rectification(M1, M2, K1, K2):
    p1_old = K1@M1
    p2_old = K2@M2
    k_new = (K1+K2)/2

    c1 = np.linalg.inv(M1)@np.array([0, 0, 0, 1]).T
    c2 = np.linalg.inv(M2)@np.array([0, 0, 0, 1]).T
    b = (c2 - c1)[:3]
    vx = b / np.sqrt(np.sum(b**2))

    vy = np.cross(M1[2][:3], vx)
    vy = vy/np.sqrt(np.sum(vy**2))

    vz = np.cross(vx, vy)
    vz = vz / np.sqrt(np.sum(vz ** 2))

    R_new = np.array([vx, vy, vz])


    M1_new = M1.copy()
    M1_new[:3, :3] = R_new[:3, :3]

    M2_new = M2.copy()
    M2_new[:3, :3] = R_new[:3, :3]

    p1_new = k_new @ M1_new
    p2_new = k_new @ M2_new

    H1 = p1_new[:3, :3] @ np.linalg.inv(p1_old[:3, :3])
    H2 = p2_new[:3, :3] @ np.linalg.inv(p2_old[:3, :3])
    return H1, H2, b


