import numpy as np
import cv2

def get_surrounding_pixels(corner_tuple, size=5):
    v = corner_tuple[0].reshape((-1, 1))
    out_v = (v + np.append(np.arange(size+1), -np.arange(1, size+1))).repeat(2*size+1, axis=1)
    u = corner_tuple[1].reshape((-1, 1))
    out_u = np.tile(u + np.append(np.arange(size+1), -np.arange(1, size+1)), 2*size+1)
    return out_v, out_u

C_val = 0.03
p = 0.5
r = 10
d = 250
dsq = d * d
t = np.cos(30/180*np.pi)

img = cv2.imread("pattern.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=5)  # Equivalent to gaussian blur and then derivative
ry = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=5)  # Equivalent to gaussian blur and then derivative
rxx = cv2.Sobel(rx, cv2.CV_32F, 1, 0, ksize=1)
rxy = cv2.Sobel(rx, cv2.CV_32F, 0, 1, ksize=1)
ryy = cv2.Sobel(ry, cv2.CV_32F, 0, 1, ksize=1) + np.random.uniform(-0.001, 0.001, rxx.shape)

C_1 = rxx + ryy
C_2 = np.sqrt(np.square(rxx-ryy)+4*np.square(rxy))

lambda1 = 0.5 * (C_1 + C_2)
lambda2 = 0.5 * (C_1 - C_2)
epsilon = C_val * np.max(lambda1)

corner_inc = np.where(np.logical_and(lambda1 > epsilon, lambda2 < -epsilon))
border_constraint = np.where(np.logical_and(np.logical_and(corner_inc[0] > r, corner_inc[0] < len(img_gray) - r),
                                            np.logical_and(corner_inc[1] > r, corner_inc[1] < len(img_gray[0]) - r)))
corner_inc = (corner_inc[0][border_constraint],
              corner_inc[1][border_constraint])


surrounding_i = get_surrounding_pixels(corner_inc, 4)

S = cv2.GaussianBlur((rxx*ryy - rxy*rxy).astype(np.float32), (9, 9), 3)[surrounding_i]
curr_corners = corner_inc[0][np.where(np.argmin(S, axis=1) == 0)], corner_inc[1][np.where(np.argmin(S, axis=1) == 0)]


sq_dists = (np.square(curr_corners[0][None, ] - curr_corners[0][:, None]) +
            np.square(curr_corners[1][None, ] - curr_corners[1][:, None]))
smallest_dists = np.sqrt(np.partition(sq_dists, 1, axis=0)[1])


#smallest_dists = np.random.normal(3.0, 2, 10000)
hist_count, hist_buckets = np.histogram(smallest_dists, bins=int(np.max(img.shape)/10))
mean_ind = np.argmax(hist_count)
count_thresh = 0.8*len(smallest_dists)
upper_i, lower_i = mean_ind, mean_ind
for i in range(len(hist_count)):
    upper_i = np.clip(mean_ind+i, 0, len(hist_count)-1)
    lower_i = np.clip(mean_ind-i, 0, len(hist_count)-1)
    count_in_r = np.sum(hist_count[lower_i: upper_i+1])
    if count_in_r >= count_thresh:
        break

left_bracket = hist_buckets[lower_i]
right_bucket = hist_buckets[upper_i+1]
dist_in_bracket = smallest_dists[np.where(np.logical_and(smallest_dists >= left_bracket,
                                                         smallest_dists <= right_bucket))]
mean_dist = np.mean(dist_in_bracket)
std_dist = np.std(dist_in_bracket)

a_min = mean_dist - 3*std_dist
a_max = mean_dist + 3*std_dist

r = 0.7*a_min
p = 0.3*a_max/a_min
d = 2*a_max
t = 0.4*a_max/a_min

