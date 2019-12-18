import numpy as np
import cv2


img = cv2.imread("p2.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_ind_circles(r=10):
    if get_ind_circles.r_prev == r:
        return (get_ind_circles.ind_in_d_1, get_ind_circles.ind_in_d_2,
                get_ind_circles.ind_in_d_3, get_ind_circles.ind_in_d_4,
                get_ind_circles.ind_in_d_5, get_ind_circles.ind_in_d_6,
                get_ind_circles.ind_in_d_7, get_ind_circles.ind_in_d_8)
    square_ind = np.indices((2 * r + 1, 2 * r + 1)) - r

    d1 = 0
    d2 = -45
    d3 = -90
    d4 = -135
    d5 = 135
    d6 = 90
    d7 = 45

    dist = np.sum(np.square(square_ind), axis=0)

    angles = 180 * (np.arctan2(square_ind[0], square_ind[1]))

    ind_in_d_1 = np.where(
        np.logical_and(dist <= r * r,
                       np.logical_and(angles <= np.pi * d1, angles > np.pi * d2))
    )
    ind_in_d_2 = np.where(
        np.logical_and(dist <= r * r,
                       np.logical_and(angles <= np.pi * d2, angles > np.pi * d3))
    )
    ind_in_d_3 = np.where(
        np.logical_and(dist <= r * r,
                       np.logical_and(angles <= np.pi * d3, angles > np.pi * d4))
    )
    ind_in_d_4 = np.where(
        np.logical_and(dist <= r * r, angles <= np.pi * d4)
    )
    ind_in_d_5 = np.where(
        np.logical_and(dist <= r * r, angles > np.pi * d5)
    )
    ind_in_d_6 = np.where(
        np.logical_and(dist <= r * r,
                       np.logical_and(angles <= np.pi * d5, angles > np.pi * d6))
    )
    ind_in_d_7 = np.where(
        np.logical_and(dist <= r * r,
                       np.logical_and(angles <= np.pi * d6, angles > np.pi * d7))
    )
    ind_in_d_8 = np.where(
        np.logical_and(dist <= r * r,
                       np.logical_and(angles <= np.pi * d7, angles > np.pi * d1))
    )
    get_ind_circles.ind_in_d_1 = np.append(square_ind[0][ind_in_d_1][None, None], square_ind[1][ind_in_d_1][None, None],
                                           axis=0)
    get_ind_circles.ind_in_d_2 = np.append(square_ind[0][ind_in_d_2][None, None], square_ind[1][ind_in_d_2][None, None],
                                           axis=0)
    get_ind_circles.ind_in_d_3 = np.append(square_ind[0][ind_in_d_3][None, None], square_ind[1][ind_in_d_3][None, None],
                                           axis=0)
    get_ind_circles.ind_in_d_4 = np.append(square_ind[0][ind_in_d_4][None, None], square_ind[1][ind_in_d_4][None, None],
                                           axis=0)
    get_ind_circles.ind_in_d_5 = np.append(square_ind[0][ind_in_d_5][None, None], square_ind[1][ind_in_d_5][None, None],
                                           axis=0)
    get_ind_circles.ind_in_d_6 = np.append(square_ind[0][ind_in_d_6][None, None], square_ind[1][ind_in_d_6][None, None],
                                           axis=0)
    get_ind_circles.ind_in_d_7 = np.append(square_ind[0][ind_in_d_7][None, None], square_ind[1][ind_in_d_7][None, None],
                                           axis=0)
    get_ind_circles.ind_in_d_8 = np.append(square_ind[0][ind_in_d_8][None, None], square_ind[1][ind_in_d_8][None, None],
                                           axis=0)
    return (get_ind_circles.ind_in_d_1, get_ind_circles.ind_in_d_2,
            get_ind_circles.ind_in_d_3, get_ind_circles.ind_in_d_4,
            get_ind_circles.ind_in_d_5, get_ind_circles.ind_in_d_6,
            get_ind_circles.ind_in_d_7, get_ind_circles.ind_in_d_8)


get_ind_circles.r_prev = None
get_ind_circles.ind_in_d_1 = None
get_ind_circles.ind_in_d_2 = None
get_ind_circles.ind_in_d_3 = None
get_ind_circles.ind_in_d_4 = None
get_ind_circles.ind_in_d_5 = None
get_ind_circles.ind_in_d_6 = None
get_ind_circles.ind_in_d_7 = None
get_ind_circles.ind_in_d_8 = None


def apply_circular_mask_and_get_averages(local_min_i, r=10):
    i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8 = get_ind_circles(r)

    local_min_i_array = np.array(local_min_i)[:, :, None]
    ind_i_1 = local_min_i_array + i_1
    ind_i_2 = local_min_i_array + i_2
    ind_i_3 = local_min_i_array + i_3
    ind_i_4 = local_min_i_array + i_4
    ind_i_5 = local_min_i_array + i_5
    ind_i_6 = local_min_i_array + i_6
    ind_i_7 = local_min_i_array + i_7
    ind_i_8 = local_min_i_array + i_8

    return (np.average(img_gray[ind_i_1[0], ind_i_1[1]], axis=1), np.average(img_gray[ind_i_2[0], ind_i_2[1]], axis=1),
            np.average(img_gray[ind_i_3[0], ind_i_3[1]], axis=1), np.average(img_gray[ind_i_4[0], ind_i_4[1]], axis=1),
            np.average(img_gray[ind_i_5[0], ind_i_5[1]], axis=1), np.average(img_gray[ind_i_6[0], ind_i_6[1]], axis=1),
            np.average(img_gray[ind_i_7[0], ind_i_7[1]], axis=1), np.average(img_gray[ind_i_8[0], ind_i_8[1]], axis=1))



local_min_i = (np.array([89, 89, 89], dtype=np.int64), np.array([83, 249, 415], dtype=np.int64))

i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8 = get_ind_circles(10)

local_min_i_array = np.array(local_min_i)[:, :, None]
ind_i_1 = local_min_i_array + i_1
ind_i_2 = local_min_i_array + i_2
ind_i_3 = local_min_i_array + i_3
ind_i_4 = local_min_i_array + i_4
ind_i_5 = local_min_i_array + i_5
ind_i_6 = local_min_i_array + i_6
ind_i_7 = local_min_i_array + i_7
ind_i_8 = local_min_i_array + i_8


avg_1 = np.average(img_gray[ind_i_1[0], ind_i_1[1]], axis=1)
avg_2 = np.average(img_gray[ind_i_2[0], ind_i_2[1]], axis=1)
avg_3 = np.average(img_gray[ind_i_3[0], ind_i_3[1]], axis=1)
avg_4 = np.average(img_gray[ind_i_4[0], ind_i_4[1]], axis=1)
avg_5 = np.average(img_gray[ind_i_5[0], ind_i_5[1]], axis=1)
avg_6 = np.average(img_gray[ind_i_6[0], ind_i_6[1]], axis=1)
avg_7 = np.average(img_gray[ind_i_7[0], ind_i_7[1]], axis=1)
avg_8 = np.average(img_gray[ind_i_8[0], ind_i_8[1]], axis=1)

img[ind_i_1[0].flat, ind_i_1[1].flat] = (0,0,255)
img[ind_i_2[0].flat, ind_i_2[1].flat] = (0,255,0)
img[ind_i_3[0].flat, ind_i_3[1].flat] = (255,0,0)
img[ind_i_4[0].flat, ind_i_4[1].flat] = (255,0,255)
img[ind_i_5[0].flat, ind_i_5[1].flat] = (0,255,255)
img[ind_i_6[0].flat, ind_i_6[1].flat] = (255,255,0)
img[ind_i_7[0].flat, ind_i_7[1].flat] = (100,100,25)
img[ind_i_8[0].flat, ind_i_8[1].flat] = (25,100,25)

#selected_region[ind_in_d_8] = 255

#cv2.imshow("sr", cv2.resize(selected_region, (8*10+1, 8*10+1), interpolation=cv2.INTER_NEAREST))
cv2.imshow("sr", cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

cv2.waitKey(0)