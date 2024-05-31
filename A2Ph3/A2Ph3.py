import utils
from ImgRectification import rectification, img_warp
import img_polt
from A2_P1.A2_utilities import NearestNeighbourInterpolation
from StereoMatching import stereo_match

import cv2
def image_preprocessing(data, img):
    if "k1" in data:
        img = NearestNeighbourInterpolation(img, data["k1"], data["k2"])
    if "H" in data:
        img, _ = img_warp(data["H"], img)
    return img


def main():
    left_cube_img = utils.get_rgb_image("2_distorted images/H3/Left/H3_Cube_Left.png")
    right_cube_img = utils.get_rgb_image("2_distorted images/H3/Right/H3_Cube_Right.png")

    print("calculate k1, k2, and rectification")
    data1, data2 = rectification(left_cube_img, right_cube_img, 0.000155, need_undistort=True, generate_image=False, debug=False)

    left_obj_img = utils.get_rgb_image("2_distorted images/H3/Left/left_1.png")
    right_obj_img = utils.get_rgb_image("2_distorted images/H3/Right/right_1.png")
    img_polt.plot_two_img(left_obj_img, right_obj_img, title="Row Input Image")

    print("image preprocessing, this may require long time, please wait...")
    print("0/2")
    left_obj_img = image_preprocessing(data1, left_obj_img)

    print("1/2")
    right_obj_img = image_preprocessing(data2, right_obj_img)

    print("2/2")
    img_polt.plot_two_img(left_obj_img, right_obj_img, with_line=True, title="Reprocessed Image")

    left_obj_img = cv2.cvtColor(left_obj_img, cv2.COLOR_RGB2GRAY)
    right_obj_img = cv2.cvtColor(right_obj_img, cv2.COLOR_RGB2GRAY)
    print("start stereo match...")
    stereo_match(left_obj_img, right_obj_img, disparity_max=64, score='ncc', window_size=25, resample=0.1)


if __name__ == "__main__":
    main()
