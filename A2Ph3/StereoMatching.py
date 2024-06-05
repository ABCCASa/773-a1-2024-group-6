
import numpy as np
import utils
import matplotlib.pyplot as pyplot
import cv2

def sad_stereo_match(img1, img2,disparity_min=1, disparity_max=32, size=7):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    layers = []
    mean_kernel = np.ones((size, size)) / size**2

    for i in range(disparity_min, disparity_max):
        new_layer = np.pad(img2, ((0, 0), (i, 0)), mode="edge")[:, :img1.shape[1]]
        sad = utils.convolution_2d(np.abs(new_layer - img1), mean_kernel)
        layers.append(sad)
    layers = np.array(layers)

    disp_map = np.argmin(layers, axis=0)+disparity_min

    pyplot.imshow(disp_map)
    pyplot.show()

    return disp_map


def ssd_stereo_match(img1, img2, disparity_min=1, disparity_max=32, size=7):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    layers = []
    mean_kernel = np.ones((size, size)) / size**2

    for i in range(disparity_min, disparity_max):
        new_layer = np.pad(img2, ((0, 0), (i, 0)), mode="edge")[:, :img1.shape[1]]

        ssd = utils.convolution_2d((new_layer - img1)**2, mean_kernel)
        layers.append(ssd)
    layers = np.array(layers)

    disp_map = np.argmin(layers, axis=0)+disparity_min

    pyplot.imshow(disp_map)
    pyplot.show()

    return disp_map


def ncc_stereo_match(img1, img2,disparity_min=1, disparity_max=32, size=31, smooth=None):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    pixel_count = size**2
    layers = []
    sum_kernel = np.ones((size, size))
    img1_sum = utils.convolution_2d(img1, sum_kernel)
    img1_mean = img1_sum / pixel_count
    img1_sub_mean_sum = utils.convolution_2d(img1**2, sum_kernel) + img1_mean**2 * pixel_count - 2 * img1_sum * img1_mean

    current = 0
    total = disparity_max - disparity_min
    for i in range(disparity_min, disparity_max):
        new_layer = np.pad(img2, ((0, 0), (i, 0)), mode="edge")[:, :img1.shape[1]]
        layer_sum = utils.convolution_2d(new_layer, sum_kernel)
        layer_mean = layer_sum / pixel_count
        layer_sub_mean_sum = utils.convolution_2d(new_layer ** 2, sum_kernel) + layer_mean ** 2 * pixel_count - 2 * layer_sum * layer_mean
        bottom = np.sqrt(layer_sub_mean_sum * img1_sub_mean_sum)
        bottom[bottom == 0] = 1
        top = utils.convolution_2d(img1*new_layer, sum_kernel) + (layer_mean*img1_mean)*pixel_count - layer_sum * img1_mean - img1_sum * layer_mean
        ncc = ((top / bottom)+1)/2
        current+=1
        layers.append(ncc)
        print(f"{current}/{total}")

    layers = np.array(layers)

    if smooth is None:
        disp_map = np.argmax(layers, axis=0)
    else:
        disp_map = np.zeros(img1.shape)
        last_max_arg = np.argmax(layers[:, :, 0], axis=0)
        disp_map[:, 0] = last_max_arg
        for i in range(1, img1.shape[1]):
            layers[last_max_arg, np.arange(img1.shape[0]), i] += smooth
            last_max_arg = np.argmax(layers[:, :, i], axis=0)
            disp_map[:, i] = last_max_arg

    disp_map += disparity_min


    pyplot.imshow(disp_map, cmap= "jet")
    pyplot.show()

    return disp_map



def stereo_match(img1, img2,disparity_min=1, disparity_max=32, window_size=31, score="ncc", smooth=None,resample = None):
    if img1.shape != img2.shape:
        raise ValueError("two image with different size")

    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    if len(img1.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    if resample != None:
        new_size = (int(img1.shape[1]*resample), int(img1.shape[0]*resample))
        img1 = cv2.resize(img1,  new_size)
        img2 = cv2.resize(img2, new_size)


    if score == "ncc":
        ncc_stereo_match(img1, img2, disparity_min, disparity_max, window_size,smooth)
    elif score == "sad":
        sad_stereo_match(img1, img2, disparity_min, disparity_max, window_size)
    elif score == "ssd":
        ssd_stereo_match(img1, img2, disparity_min, disparity_max, window_size)
    else:
        raise ValueError("score can only be 'ncc', 'sad', 'ssd'")

def main():
    left_obj_img = utils.get_gray_image("own images/Rocks2/view0.png")
    right_obj_img = utils.get_gray_image("own images/Rocks2/view1.png")


    stereo_match(left_obj_img, right_obj_img, disparity_max=20, score='ncc', window_size=41)



if __name__ == "__main__":
    main()
