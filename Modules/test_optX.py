import numpy as np
import cv2
from scipy.optimize import root as optAlg

dta = np.load("XandxandHs.npz")
X = dta['arr_0']
x = dta['arr_1']
Hs = dta['arr_2']


h_X = np.empty((3, X.shape[1]))
h_X[:2, :] = X[0].T
h_X[2, :] = 1.0

def getError(H_vals, X, x):
    H = H_vals.reshape(3,3)
    hold = np.dot(H, h_X)
    return np.square(x-(hold[:2] / hold[2]).T).flatten()


def getJ(H_vals, X, _):
    H = H_vals.reshape(3,3)
    proj = np.dot(H, h_X)
    sw = 1/proj[2]
    sw2 = -1/(proj[2]*proj[2])
    swx = proj[0]*sw2
    swy = proj[1]*sw2
    zeros = np.zeros(X.shape[1])
    J = np.empty((9, 2*X.shape[1]))
    J[:, 0::2] = np.stack((X[0]*sw, X[1]*sw, sw, zeros, zeros, zeros, X[0]*swx, X[1]*swx, swx), axis=0)
    J[:, 1::2] = np.stack((zeros, zeros, zeros, X[0]*sw, X[1]*sw, sw, X[0]*swy, X[1]*swy, swy), axis=0)
    return J.T.reshape(2*X.shape[1], -1)

Hsbefore = Hs.copy()
for i in range(len(Hs)):
    ret = optAlg(
            getError,
            Hs[i].flatten(),
            jac=getJ,
            args=(h_X, x[i]),
            method='lm'
        )
    if ret.success:
        Hs[i] = ret.x.reshape(3,3)/ret.x[8]




"""
import numpy as np
import cv2
from scipy.optimize import root as optAlg

dta = np.load("XandxandHs.npz")
X = dta['arr_0']
x = dta['arr_1']
Hs = dta['arr_2']


h_X = np.empty((3, X.shape[1]))
h_X[:2, :] = X[0].T
h_X[2, :] = 1.0
np.einsum("ijk,kl->ilj", Hs, h_X)
stackedX = np.tile(h_X[:2][:, None], (1, x.shape[0], 1))

def getError(H_vals, X, x, _):
    H = H_vals.reshape(-1,3,3)
    hold = np.einsum("ijk,kl->ilj", H, h_X)
    return (x-(hold[:, :, :2] / hold[:, :, 2][:, :, None])).flatten()


def getJ(H_vals, X, _, stackedX):
    H = H_vals.reshape(-1,3,3)
    proj = np.einsum("ijk,kl->ilj", H, X)
    sw = 1/proj[:, :, 2]
    sw2 = -1/(proj[:, :, 2]*proj[:, :, 2])
    swx = proj[:, :, 0]*sw2
    swy = proj[:, :, 1]*sw2
    zeros = np.zeros((H.shape[0], X.shape[1]))
    J = np.empty((9, H.shape[0], 2*X.shape[1]))
    J[:, :, 0::2] = np.stack((stackedX[0]*sw, stackedX[1]*sw, sw, zeros, zeros, zeros, stackedX[0]*swx, stackedX[1]*swx, swx), axis=0)
    J[:, :, 1::2] = np.stack((zeros, zeros, zeros, stackedX[0]*sw, stackedX[1]*sw, sw, stackedX[0]*swy, stackedX[1]*swy, swy), axis=0)
    return J.T.reshape(2*X.shape[1], -1).astype(float)


ret = optAlg(
        getError,
        Hs.flatten(),
        jac=getJ,
        args=(h_X, x, stackedX),
        method='lm'
    )
"""