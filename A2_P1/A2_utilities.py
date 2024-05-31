import numpy as np
import cv2
import A2_P1.plot_image
def NearestNeighbourInterpolation(img,k1, k2):

    def undistPixel(xd,yd,k1, k2):
        cx=img.shape[1]//2
        cy=img.shape[0]//2
        xu = (xd-cx)*(1+k1*((xd-cx)**2+(yd-cy)**2)+k2*((xd-cx)**4+(yd-cy)**4))+cx
        yu = (yd-cy)*(1+k1*((xd-cx)**2+(yd-cy)**2)+k2*((xd-cx)**4+(yd-cy)**4))+cy
        return xu, yu

    #Find known points
    known = dict()
    out = -1*np.ones(img.shape)
    for y in range(0, img.shape[0]*2):
        for x in range(0, img.shape[1]*2):
            x_real = x//2
            y_real = y//2
            xu, yu = undistPixel(x / 2, y / 2, k1, k2)
            xu_int = int(round(xu))
            yu_int = int(round(yu))

            #If value is within the image bounds
            if 0<= int(round(xu)) < img.shape[1] and 0 <= int(round(yu)) < img.shape[0]:
                #Find closest matching point
                if (xu_int, yu_int) not in known:
                    known[(xu_int, yu_int)] = (xu_int-xu)**2+(yu_int-yu)**2
                    out[yu_int,xu_int] = img[y_real,x_real]
                else:
                    #If a new point is found closer to the original
                    if known[(xu_int, yu_int)] > (xu_int-xu)**2+(yu_int-yu)**2:
                        known[(xu_int, yu_int)] = (xu_int-xu)**2+(yu_int-yu)**2
                        out[yu_int,xu_int] = img[y_real,x_real]

    out = out.astype(np.uint8)
    return out


def show(img, scale_percent = 30, waitKey=-1):
    w = int(img.shape[1] * scale_percent / 100)
    h = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
    cv2.imshow('image', img)
    k = cv2.waitKey(waitKey) & 0xFF
    if k == ord('s'):
        cv2.imwrite("image.png", img)
        cv2.destroyAllWindows()
    if k == ord('q'):
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()
