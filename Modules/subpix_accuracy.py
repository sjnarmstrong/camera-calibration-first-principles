import cv2
import numpy as np


def get_surrounding_pixels(corner_tuple, size=1):
    v = corner_tuple[0].reshape((-1, 1))
    out_v = (v + np.append(np.arange(size + 1), -np.arange(1, size + 1))).repeat(2 * size + 1, axis=1)
    u = corner_tuple[1].reshape((-1, 1))
    out_u = np.tile(u + np.append(np.arange(size + 1), -np.arange(1, size + 1)), 2 * size + 1)
    return out_v, out_u


img = cv2.imread("pattern.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = np.array([[184, 184, 184, 184, 184, 184, 184, 184, 184, 350, 350,
                     350, 350, 350, 350, 350, 350, 350, 516, 516, 516, 516,
                     516, 516, 516, 516, 516, 683, 683, 683, 683, 683, 683,
                     683, 683, 683, 849, 849, 849, 849, 849, 849, 849, 849,
                     849, 1015, 1015, 1015, 1015, 1015, 1015, 1015, 1015, 1015],
                    [184, 350, 516, 683, 849, 1015, 1182, 1348, 1514, 184, 350,
                     516, 683, 849, 1015, 1182, 1348, 1514, 184, 350, 516, 683,
                     849, 1015, 1182, 1348, 1514, 184, 350, 516, 683, 849, 1015,
                     1182, 1348, 1514, 184, 350, 516, 683, 849, 1015, 1182, 1348,
                     1514, 184, 350, 516, 683, 849, 1015, 1182, 1348, 1514]],
                   dtype=np.int64)
surrounding_i = get_surrounding_pixels(corners.astype(np.int64), 2)
Isqr = img_gray[surrounding_i]
Isqr *= Isqr
C = np.sum(Isqr, axis=1)

y_subpix = np.sum(surrounding_i[0]*Isqr, axis=1)/C
x_subpix = np.sum(surrounding_i[1]*Isqr, axis=1)/C
