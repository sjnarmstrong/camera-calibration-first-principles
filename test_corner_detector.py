import os
from os import listdir
from os.path import splitext

import cv2
import numpy as np
from Modules.CornerDetector import detect_checkerboard_corners
from Modules.ZhangAlgV2 import get_camera_calib_m

templateMat = r"""
\begin{{pmatrix}}
{0}&{1}&{2}\\0&{4}&{5}\\0&0&1
\end{{pmatrix}}
"""

dataset_names = ["dataset1/", "dataset2/", "dataset3/", "dataset4/", "dataset5/", "dataset6/", "dataset7/"]
dataset_shapes = [(20, 16), (13, 12), (8, 6), (16, 16), (12, 8), (11, 8), (9, 7), (16, 16)]
c_vals = [0.18, 0.24, 0.18, 0.1, 0.15, 0.1, 0.18, 0.18]
base_path = "./Datasets/calibration_datasets/"
base_path_out = "./Outputs/calibration_datasets/"

for i, datasetName in enumerate(dataset_names):
    dataset_path = base_path+datasetName
    dataset_shape_cv = dataset_shapes[i]
    dataset_shape = dataset_shape_cv[::-1]
    print(datasetName+"___________________________")
    correct_count = 0
    cv_correct_count = 0
    count = 0
    all_detected_points_mine = []
    all_detected_points_cv = []
    wpcv=[]
    wpmine=[]

    indy, indx = np.indices(dataset_shape)
    world_pts = np.vstack((indx.flat, indy.flat, np.zeros(indx.size))).T.astype(np.float32)
    for img_name in listdir(dataset_path):
        img = cv2.imread(dataset_path+img_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_successfully_cv, ordered_points_cv = cv2.findChessboardCorners(img_gray, dataset_shape_cv)
        ordered_points, detected_successfully = detect_checkerboard_corners(img_gray, expected_shape=dataset_shape, C_val=c_vals[i], max_iter=10)

        ordered_points[[1, 0]] = ordered_points[[0, 1]]
        ordered_points = ordered_points.T[:, None, :].astype(np.float32)

        if detected_successfully:
            all_detected_points_mine.append(ordered_points)
            wpmine.append(world_pts)
        if detected_successfully_cv:
            all_detected_points_cv.append(ordered_points_cv)
            wpcv.append(world_pts)
        count += 1
        correct_count += detected_successfully
        cv_correct_count += detected_successfully_cv

        cb_corners_img = cv2.drawChessboardCorners(img, dataset_shape_cv, ordered_points, detected_successfully)
        os.makedirs(base_path_out+datasetName, exist_ok=True)
        cv2.imwrite(base_path_out+datasetName+splitext(img_name)[0]+".png", cb_corners_img)
    print(correct_count,cv_correct_count,count)
    if cv_correct_count>0:
        print("CV-CV")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(wpcv, all_detected_points_cv, img_gray.shape[::-1], None, (0, 0, 0, 0, 0),
                                                           flags=cv2.CALIB_FIX_K1 +
                                                                 cv2.CALIB_FIX_K2 +
                                                                 cv2.CALIB_FIX_K3 +
                                                                 cv2.CALIB_FIX_K4 +
                                                                 cv2.CALIB_FIX_K5 +
                                                                 cv2.CALIB_FIX_K6 + cv2.CALIB_FIX_TANGENT_DIST)
        print(mtx)
        print(templateMat.format(int(np.rint(mtx[0, 0])),
                                 int(np.rint(mtx[0, 1])),
                                 int(np.rint(mtx[0, 2])),
                                 int(np.rint(mtx[1, 0])),
                                 int(np.rint(mtx[1, 1])),
                                 int(np.rint(mtx[1, 2]))))
        print("CV-Mine")
        mtx = get_camera_calib_m(world_pts[None,:,:2], np.array(all_detected_points_cv).squeeze(axis=2))
        print(mtx)
        print(templateMat.format(int(np.rint(mtx[0, 0])),
                                 int(np.rint(mtx[0, 1])),
                                 int(np.rint(mtx[0, 2])),
                                 int(np.rint(mtx[1, 0])),
                                 int(np.rint(mtx[1, 1])),
                                 int(np.rint(mtx[1, 2]))))
    if correct_count>0:
        print("Mine-CV")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(wpmine, all_detected_points_mine, img_gray.shape[::-1], None, (0, 0, 0, 0, 0),
                                                           flags=cv2.CALIB_FIX_K1 +
                                                                 cv2.CALIB_FIX_K2 +
                                                                 cv2.CALIB_FIX_K3 +
                                                                 cv2.CALIB_FIX_K4 +
                                                                 cv2.CALIB_FIX_K5 +
                                                                 cv2.CALIB_FIX_K6 + cv2.CALIB_FIX_TANGENT_DIST)
        print(mtx)
        print(templateMat.format(int(np.rint(mtx[0, 0])),
                                 int(np.rint(mtx[0, 1])),
                                 int(np.rint(mtx[0, 2])),
                                 int(np.rint(mtx[1, 0])),
                                 int(np.rint(mtx[1, 1])),
                                 int(np.rint(mtx[1, 2]))))
        print("Mine-Mine")
        mtx = get_camera_calib_m(world_pts[None,:,:2], np.array(all_detected_points_cv).squeeze(axis=2))
        print(mtx)
        print(templateMat.format(int(np.rint(mtx[0, 0])),
                                 int(np.rint(mtx[0, 1])),
                                 int(np.rint(mtx[0, 2])),
                                 int(np.rint(mtx[1, 0])),
                                 int(np.rint(mtx[1, 1])),
                                 int(np.rint(mtx[1, 2]))))

