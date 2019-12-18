real_points = (np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
                         [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
                         [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7],
                         [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7],
                         [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7]]) * [15.0, 15.0]).T


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