import numpy as np

local_min_i = (np.array([89, 89, 100, 101, 102, 89,1000,1000,1000,1000]), np.array([83, 249, 250, 251, 252, 415,1001,1002,1003,1004]))

d = 122
dsq = d * d
sq_dists = (np.square(local_min_i[0][None, ] - local_min_i[0][:, None]) +
            np.square(local_min_i[1][None, ] - local_min_i[1][:, None]))

ind_dist_constraint = np.where(np.count_nonzero(sq_dists < dsq, axis=0) > 3)
local_min_i = local_min_i[0][ind_dist_constraint], local_min_i[1][ind_dist_constraint]
sq_dists = sq_dists[ind_dist_constraint].T[ind_dist_constraint].T


t = np.cos(30/180*np.pi)
max_it = 1000

for it in range(max_it):
    if len(local_min_i[0]) < 3:
        break
    local_min_i_arr = np.array(local_min_i)
    k_smallest_dists = np.argpartition(sq_dists, 2, axis=0)
    points_1 = local_min_i_arr[:, k_smallest_dists[np.where(np.choose(k_smallest_dists[:2], sq_dists) > 0)]]
    points_2 = local_min_i_arr[:, k_smallest_dists[2]]
    vector_1 = points_1 - local_min_i_arr
    vector_2 = points_2 - local_min_i_arr
    cos_theta = np.sum(vector_1*vector_2, axis=0)/(np.linalg.norm(vector_1, axis=0)*np.linalg.norm(vector_2, axis=0))
    ind_ang_constraint = np.where(cos_theta < t)

    if len(local_min_i[0]) == len(ind_ang_constraint):
        break

    local_min_i = local_min_i[0][ind_ang_constraint], local_min_i[1][ind_ang_constraint]
    sq_dists = sq_dists[ind_ang_constraint].T[ind_ang_constraint].T
#k_smallest_dists = np.argpartition(sq_dists, 3, axis=0)
#ind_dist_constraint = np.where(np.choose(k_smallest_dists[3], sq_dists) < dsq)
#local_min_i = local_min_i[0][ind_dist_constraint], local_min_i[1][ind_dist_constraint]
#k_smallest_dists = k_smallest_dists[ind_dist_constraint].T[ind_dist_constraint].T
#sq_dists = sq_dists[ind_dist_constraint].T[ind_dist_constraint].T


#ind_sorted = np.argsort(np.choose(k_smallest_dists[:3], sq_dists), axis=0)
#local_min_i
