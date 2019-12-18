import numpy as np
import cv2


def get_surrounding_pixels(corner_tuple, size=1):
    v = corner_tuple[0].reshape((-1, 1))
    out_v = (v + np.append(np.arange(size+1), -np.arange(1, size+1))).repeat(2*size+1, axis=1)
    u = corner_tuple[1].reshape((-1, 1))
    out_u = np.tile(u + np.append(np.arange(size+1), -np.arange(1, size+1)), 2*size+1)
    return out_v, out_u


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


def apply_circular_mask_and_d_values(indices_to_check, p, img_gray, r=10):
    i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8 = get_ind_circles(r)

    min_ind = [[[0]], [[0]]]
    max_ind = [[[img_gray.shape[0]-1]], [[img_gray.shape[1]-1]]]

    local_min_i_array = np.array(indices_to_check)[:, :, None]
    ind_i_1 = np.clip(local_min_i_array + i_1, min_ind, max_ind)
    ind_i_2 = np.clip(local_min_i_array + i_2, min_ind, max_ind)
    ind_i_3 = np.clip(local_min_i_array + i_3, min_ind, max_ind)
    ind_i_4 = np.clip(local_min_i_array + i_4, min_ind, max_ind)
    ind_i_5 = np.clip(local_min_i_array + i_5, min_ind, max_ind)
    ind_i_6 = np.clip(local_min_i_array + i_6, min_ind, max_ind)
    ind_i_7 = np.clip(local_min_i_array + i_7, min_ind, max_ind)
    ind_i_8 = np.clip(local_min_i_array + i_8, min_ind, max_ind)

    avg_1 = np.average(img_gray[ind_i_1[0], ind_i_1[1]], axis=1)
    avg_2 = np.average(img_gray[ind_i_2[0], ind_i_2[1]], axis=1)
    avg_3 = np.average(img_gray[ind_i_3[0], ind_i_3[1]], axis=1)
    avg_4 = np.average(img_gray[ind_i_4[0], ind_i_4[1]], axis=1)
    avg_5 = np.average(img_gray[ind_i_5[0], ind_i_5[1]], axis=1)
    avg_6 = np.average(img_gray[ind_i_6[0], ind_i_6[1]], axis=1)
    avg_7 = np.average(img_gray[ind_i_7[0], ind_i_7[1]], axis=1)
    avg_8 = np.average(img_gray[ind_i_8[0], ind_i_8[1]], axis=1)

    D_1 = np.abs(avg_1-avg_5)
    D_2 = np.abs(avg_3-avg_7)
    D_3 = np.abs(avg_1+avg_5-avg_3-avg_7)/2.0
    D_4 = np.abs(avg_2-avg_6)
    D_5 = np.abs(avg_4-avg_8)
    D_6 = np.abs(avg_2+avg_6-avg_4-avg_8)/2.0
    return D_1, D_2, p*D_3, D_4, D_5, p*D_6


def apply_centrosymmetry(curr_corners, p, r, img_gray):
    D_1, D_2, pD_3, D_4, D_5, pD_6 = apply_circular_mask_and_d_values(curr_corners, p, img_gray, r)
    centro_sym_ind = np.where(np.logical_or(np.logical_and(D_1 < pD_3, D_2 < pD_3),
                                            np.logical_and(D_4 < pD_6, D_5 < pD_6)))

    return curr_corners[0][centro_sym_ind], curr_corners[1][centro_sym_ind]


def apply_dist_constraint(curr_corners, d_sq):
    if len(curr_corners[0]) < 3:
        return [], []

    sq_dists = (np.square(curr_corners[0][None, ] - curr_corners[0][:, None]) +
                np.square(curr_corners[1][None, ] - curr_corners[1][:, None]))

    ind_dist_constraint = np.where(np.count_nonzero(sq_dists < d_sq, axis=0) > 3)
    return (curr_corners[0][ind_dist_constraint], curr_corners[1][ind_dist_constraint]), \
           sq_dists[ind_dist_constraint].T[ind_dist_constraint].T


def apply_angle_constraint(curr_corners, sq_dists, t, max_it=1000):
    for it in range(max_it):
        if len(curr_corners[0]) < 3:
            return ([], []), sq_dists
        local_min_i_arr = np.array(curr_corners)
        k_smallest_dists = np.argpartition(sq_dists, 2, axis=0)
        points_1 = local_min_i_arr[:, k_smallest_dists[
                                          np.where(sq_dists[np.arange(len(sq_dists)), k_smallest_dists[:2]] > 0)]]
        points_2 = local_min_i_arr[:, k_smallest_dists[2]]
        vector_1 = points_1 - local_min_i_arr
        vector_2 = points_2 - local_min_i_arr
        cos_theta = np.sum(vector_1 * vector_2, axis=0) / (
                    np.linalg.norm(vector_1, axis=0) * np.linalg.norm(vector_2, axis=0))
        ind_ang_constraint = np.where(cos_theta < t)

        if len(curr_corners[0]) == len(ind_ang_constraint[0]):
            return curr_corners, sq_dists

        curr_corners = curr_corners[0][ind_ang_constraint], curr_corners[1][ind_ang_constraint]
        sq_dists = sq_dists[ind_ang_constraint].T[ind_ang_constraint].T
    return curr_corners, sq_dists


def calculate_parameters(curr_corners, iter, iter_max, img_gray_shape):
    sq_dists = (np.square(curr_corners[0][None, ] - curr_corners[0][:, None]) +
                np.square(curr_corners[1][None, ] - curr_corners[1][:, None]))
    smallest_dists = np.sqrt(np.partition(sq_dists, 1, axis=0)[1])

    hist_count, hist_buckets = np.histogram(smallest_dists, bins=50)
    mean_ind = np.argmax(hist_count)
    count_thresh = 0.95 * len(smallest_dists)
    upper_i, lower_i = mean_ind, mean_ind
    for i in range(len(hist_count)):
        upper_i = np.clip(mean_ind + i, 0, len(hist_count) - 1)
        lower_i = np.clip(mean_ind - i, 0, len(hist_count) - 1)
        count_in_r = np.sum(hist_count[lower_i: upper_i + 1])
        if count_in_r >= count_thresh:
            break

    left_bracket = hist_buckets[lower_i]
    right_bucket = hist_buckets[upper_i + 1]
    dist_in_bracket = smallest_dists[np.where(np.logical_and(smallest_dists >= left_bracket,
                                                             smallest_dists <= right_bucket))]
    mean_dist = np.mean(dist_in_bracket)
    std_dist = np.std(dist_in_bracket)

    scale = 3 * (iter_max - iter)/iter_max
    a_min = np.maximum(10, mean_dist - scale * std_dist)
    a_max = np.minimum(np.max(img_gray_shape)/5, mean_dist + scale * std_dist)

    r = 0.5 * a_min
    p = 0.35 * a_max / a_min
    d = 2 * a_max
    t = 0.4 * a_max / a_min
    #print(r,p,d,t)
    return r, p, d*d, t


def get_ordered_points_between_lines(curr_corners, pt_1, pt_2, sq_dists, dist_thresh):
    pt_1_y = curr_corners[0][pt_1]
    pt_1_x = curr_corners[1][pt_1]
    pt_2_y = curr_corners[0][pt_2]
    pt_2_x = curr_corners[1][pt_2]

    dy = pt_2_y - pt_1_y
    dx = pt_2_x - pt_1_x
    x2y1_x1y2 = pt_2_x * pt_1_y - pt_1_x * pt_2_y

    dist_from_line = np.abs(dy * curr_corners[1] - dx * curr_corners[0] + x2y1_x1y2) / (np.sqrt(dy * dy + dx * dx) + 1e-9)
    ind = np.where(dist_from_line < dist_thresh)
    order = np.argsort(sq_dists[pt_1][ind])
    return ind[0][order]


def get_extreme_corners(curr_corners):
    cxpcy = curr_corners[1] + curr_corners[0]
    cxmcy = curr_corners[1] - curr_corners[0]
    return np.argmin(cxpcy), np.argmax(cxmcy), np.argmin(cxmcy), np.argmax(cxpcy)


def get_sub_pixel_accuracy(corners, img_gray):
    surrounding_i = get_surrounding_pixels(corners, 4)
    Isqr = img_gray[surrounding_i]
    Isqr *= Isqr
    C = np.sum(Isqr, axis=1)

    y_subpix = np.sum(surrounding_i[0] * Isqr, axis=1) / C
    x_subpix = np.sum(surrounding_i[1] * Isqr, axis=1) / C
    return y_subpix, x_subpix


def decrease_c(current_c, was_decreasing, current_step):
    step = current_step if was_decreasing is None or was_decreasing else current_step/10
    return current_c - step, True, step


def increase_c(current_c, was_decreasing, current_step):
    step = current_step if was_decreasing is None or not was_decreasing else current_step/10
    return current_c + step, True, step


def detect_checkerboard_corners(img_gray, expected_shape=(9, 6), C_val=0.03, max_iter=20):
    expected_corner_len = expected_shape[0]*expected_shape[1]

    scaledimg = img_gray/255
    rx = cv2.Sobel(scaledimg, cv2.CV_64F, 1, 0, ksize=7)  # Equivalent to gaussian blur and then derivative
    ry = cv2.Sobel(scaledimg, cv2.CV_64F, 0, 1, ksize=7)  # Equivalent to gaussian blur and then derivative
    rxx = cv2.Sobel(rx, cv2.CV_64F, 1, 0, ksize=1)
    rxy = cv2.Sobel(rx, cv2.CV_64F, 0, 1, ksize=1)
    ryy = cv2.Sobel(ry, cv2.CV_64F, 0, 1, ksize=1)

    C_1 = rxx + ryy
    C_2 = np.sqrt(np.square(rxx-ryy)+4*np.square(rxy))

    lambda1 = 0.5 * (C_1 + C_2)
    lambda2 = 0.5 * (C_1 - C_2)

    max_l_1 = np.max(lambda1)

    epsilon = C_val * max_l_1
    corner_inc = np.where(np.logical_and(lambda1 > epsilon, lambda2 < -epsilon))
    border_constraint = np.where(np.logical_and(np.logical_and(corner_inc[0] > 5, corner_inc[0] < len(img_gray) - 5),
                                                np.logical_and(corner_inc[1] > 5, corner_inc[1] < len(img_gray[0]) - 5)))
    corner_inc = (corner_inc[0][border_constraint],
                  corner_inc[1][border_constraint])

    surrounding_i = get_surrounding_pixels(corner_inc, 4)

    S = cv2.GaussianBlur((rxx*ryy - rxy*rxy).astype(np.float32), (9, 9), 3)[surrounding_i]
    curr_corners_i = corner_inc[0][np.where(np.argmin(S, axis=1) == 0)], corner_inc[1][np.where(np.argmin(S, axis=1) == 0)]
    if len(curr_corners_i[0]) < expected_corner_len:
        print("too few")
        return np.array(curr_corners_i), False

    r, p, dsq, t = calculate_parameters(curr_corners_i, 0, max_iter, img_gray.shape)
    r = int(r)
    for iteration in range(max_iter):
        prevLen = len(curr_corners_i[0])
        curr_corners_i = apply_centrosymmetry(curr_corners_i, p, r, img_gray)
        if len(curr_corners_i[0]) < expected_corner_len:
            print("too few")
            return np.array(curr_corners_i), False
        curr_corners_i, sq_dists = apply_dist_constraint(curr_corners_i, dsq)
        if len(curr_corners_i[0]) < expected_corner_len:
            print("too few")
            return np.array(curr_corners_i), False
        curr_corners_i, sq_dists = apply_angle_constraint(curr_corners_i, sq_dists, t)
        if len(curr_corners_i[0]) < expected_corner_len:
            print("too few")
            return np.array(curr_corners_i), False

        if len(curr_corners_i[0]) == expected_corner_len or prevLen == len(curr_corners_i[0]):
            break

        r, p, dsq, t = calculate_parameters(curr_corners_i, 0, max_iter, img_gray.shape)
        r = int(r)

    if len(curr_corners_i[0]) > expected_corner_len:
        print("Too Many")

    A, B, C, D = get_extreme_corners(curr_corners_i)

    hold = np.where(sq_dists > 0)
    alpha = 0.8
    dist_thresh = alpha * np.sqrt(np.min(sq_dists[hold]))

    edge_points_1 = get_ordered_points_between_lines(curr_corners_i, A, C, sq_dists, dist_thresh)
    edge_points_2 = get_ordered_points_between_lines(curr_corners_i, B, D, sq_dists, dist_thresh)

    if len(edge_points_1) != len(edge_points_2):
        edge_points_1 = get_ordered_points_between_lines(curr_corners_i, A, B, sq_dists, dist_thresh)
        edge_points_2 = get_ordered_points_between_lines(curr_corners_i, C, D, sq_dists, dist_thresh)

    if len(edge_points_1) != len(edge_points_2):
        return np.array(curr_corners_i), False

    ordered_points_ind = []

    for p1, p2 in zip(edge_points_1, edge_points_2):
        pts = get_ordered_points_between_lines(curr_corners_i, p1, p2, sq_dists, dist_thresh)
        ordered_points_ind.append(pts)
        if len(pts) != len(ordered_points_ind[0]):
         return np.array(curr_corners_i), False

    ordered_points_ind = np.array(ordered_points_ind)
    if expected_shape[0] != ordered_points_ind.shape[0]:
        ordered_points_ind = ordered_points_ind.T
    if expected_shape != ordered_points_ind.shape:
        return np.array(curr_corners_i), False

    curr_corners_i = get_sub_pixel_accuracy(curr_corners_i, img_gray)
    curr_corners_i = np.array(curr_corners_i)
    ordered_points = curr_corners_i.T[ordered_points_ind.flat].T
    return ordered_points, True

