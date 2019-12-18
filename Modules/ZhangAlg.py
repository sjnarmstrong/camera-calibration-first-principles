import numpy as np
import cv2

real_points = (np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
                         [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
                         [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7],
                         [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7],
                         [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7]]) * [15.0, 15.0]).T
all_detected_points = np.load("all_detected_corners.npz")['arr_0'][:, :20*real_points.shape[1]]



ScaleM = np.array([[2/640.0, 0, -1],
                   [0, 2/480.0, -1],
                   [0,       0,  1]])
iScaleM = np.linalg.inv(ScaleM)
all_detected_points_scaled = 2*all_detected_points / [[480.0], [640.0]] -1

X = np.tile(real_points, (np.array(all_detected_points_scaled.shape) / real_points.shape).astype(np.int64))

nones = -np.ones(X[1].shape)
zeros = np.zeros(X[1].shape)
Ax = np.array((-X[1], -X[0], nones, zeros, zeros, zeros, all_detected_points_scaled[1] * X[1],
               all_detected_points_scaled[1] * X[0], all_detected_points_scaled[1]))
Ay = np.array((zeros, zeros, zeros, -X[1], -X[0], nones, all_detected_points_scaled[0] * X[1],
               all_detected_points_scaled[0] * X[0], all_detected_points_scaled[0]))

M = np.empty((2 * Ax.shape[1], Ax.shape[0]), dtype=Ax.dtype)
M[0::2] = Ax.T
M[1::2] = Ay.T
M = M.reshape(-1, real_points.size, 9)

_, s_1, Vt = np.linalg.svd(M, full_matrices=False)
Hs = Vt[:, -1].reshape(-1, 3, 3)

Hs = Hs.swapaxes(1, 2)/Hs[:,2,2][:,None,None]

v_m_1_2 = np.array((
    Hs[:, 0, 0] * Hs[:, 0, 1],
    Hs[:, 0, 0] * Hs[:, 1, 1] + Hs[:, 1, 0] * Hs[:, 0, 1],
    Hs[:, 0, 0] * Hs[:, 2, 1] + Hs[:, 2, 0] * Hs[:, 0, 1],
    Hs[:, 1, 0] * Hs[:, 1, 1],
    Hs[:, 2, 0] * Hs[:, 1, 1] + Hs[:, 1, 0] * Hs[:, 2, 1],
    Hs[:, 2, 0] * Hs[:, 2, 1]))

v_m_1_1 = np.array((
    Hs[:, 0, 0] * Hs[:, 0, 0],
    Hs[:, 0, 0] * Hs[:, 1, 0] + Hs[:, 1, 0] * Hs[:, 0, 0],
    Hs[:, 0, 0] * Hs[:, 2, 0] + Hs[:, 2, 0] * Hs[:, 0, 0],
    Hs[:, 1, 0] * Hs[:, 1, 0],
    Hs[:, 2, 0] * Hs[:, 1, 0] + Hs[:, 1, 0] * Hs[:, 2, 0],
    Hs[:, 2, 0] * Hs[:, 2, 0]))

v_m_2_2 = np.array((
    Hs[:, 0, 1] * Hs[:, 0, 1],
    Hs[:, 0, 1] * Hs[:, 1, 1] + Hs[:, 1, 1] * Hs[:, 0, 1],
    Hs[:, 0, 1] * Hs[:, 2, 1] + Hs[:, 2, 1] * Hs[:, 0, 1],
    Hs[:, 1, 1] * Hs[:, 1, 1],
    Hs[:, 2, 1] * Hs[:, 1, 1] + Hs[:, 1, 1] * Hs[:, 2, 1],
    Hs[:, 2, 1] * Hs[:, 2, 1]))

V = np.empty((2 * v_m_1_2.shape[1], v_m_1_2.shape[0]), dtype=Ax.dtype)
V[0::2] = v_m_1_2.T
V[1::2] = (v_m_1_1 - v_m_2_2).T

U, S, Vt = np.linalg.svd(V, full_matrices=False)
b = Vt[-1]
B = np.array([[b[0], b[1], b[2]],
              [b[1], b[3], b[4]],
              [b[2], b[4], b[5]]])
KnT = np.linalg.cholesky(B)
K = np.linalg.inv(KnT).T
sK = np.dot(iScaleM, K/K[2,2])


all_detected_points_scaled[[1, 0]] = all_detected_points_scaled[[0, 1]]
ordered_points = all_detected_points_scaled.T.reshape(-1, 8*5, 2).astype(np.float32)
objp = np.zeros((8 * 5, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)
objp *= 15
Rx = [objp for i in range(ordered_points.shape[0])]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(Rx, ordered_points, (640, 480), None, (0, 0, 0, 0, 0), flags=cv2.CALIB_FIX_K1+
                                                                                                                cv2.CALIB_FIX_K2+
                                                                                                                cv2.CALIB_FIX_K3+
                                                                                                                cv2.CALIB_FIX_K4+
                                                                                                                cv2.CALIB_FIX_K5+
                                                                                                                cv2.CALIB_FIX_K6)
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(640, 480),1,(640, 480))
