import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import root as optAlg


def get_scale_matrix(pts, inverted_matrix=True):
    avg = np.mean(pts, axis=1)
    std = 1.4142135623730951 / np.std(pts, axis=1)
    mtx = np.zeros((std.shape[0], 3, 3))
    mtx[:, 2, 2] = 1
    if inverted_matrix:
        mtx[:, 0, 0] = 1 / std[:, 0]
        mtx[:, 1, 1] = 1 / std[:, 1]
        mtx[:, 0, 2] = avg[:, 0]
        mtx[:, 1, 2] = avg[:, 1]
    else:
        mtx[:, 0, 0] = std[:, 0]
        mtx[:, 1, 1] = std[:, 1]
        mtx[:, 0, 2] = -std[:, 0] * avg[:, 0]
        mtx[:, 1, 2] = -std[:, 1] * avg[:, 1]

    return (np.array([std[:, 0], std[:, 1]]).T[:, None, :],
            np.array([-std[:, 0] * avg[:, 0], -std[:, 1] * avg[:, 1]]).T[:, None, :],
            mtx)


def getError(H_vals, X, x):
    H = H_vals.reshape(3, 3)
    hold = np.dot(H, X)
    return np.square(x - (hold[:2] / hold[2]).T).flatten()


def getJ(H_vals, X, _):
    H = H_vals.reshape(3, 3)
    proj = np.dot(H, X)
    sw = 1 / proj[2]
    sw2 = -1 / (proj[2] * proj[2])
    swx = proj[0] * sw2
    swy = proj[1] * sw2
    zeros = np.zeros(X.shape[1])
    J = np.empty((9, 2 * X.shape[1]))
    J[:, 0::2] = np.stack((X[0] * sw, X[1] * sw, sw, zeros, zeros, zeros, X[0] * swx, X[1] * swx, swx), axis=0)
    J[:, 1::2] = np.stack((zeros, zeros, zeros, X[0] * sw, X[1] * sw, sw, X[0] * swy, X[1] * swy, swy), axis=0)
    return J.T.reshape(2 * X.shape[1], -1)


def get_camera_calib_m(X, x):
    SX, OX, MSX = get_scale_matrix(X, False)
    Sx, Ox, iMSx = get_scale_matrix(x, True)
    Nx = Sx * x + Ox
    NX = np.tile(SX * X + OX, (Nx.shape[0], 1, 1))

    nones = -np.ones(Nx.shape[:2])
    zeros = np.zeros(Nx.shape[:2])

    M = np.empty((Nx.shape[0], 2 * Nx.shape[1], 9), dtype=np.float64)
    M[:, 0::2] = np.stack((-NX[:, :, 0], -NX[:, :, 1], nones,
                           zeros, zeros, zeros,
                           NX[:, :, 0] * Nx[:, :, 0], NX[:, :, 1] * Nx[:, :, 0], Nx[:, :, 0]),
                          axis=2)
    M[:, 1::2] = np.stack((zeros, zeros, zeros,
                           -NX[:, :, 0], -NX[:, :, 1], nones,
                           NX[:, :, 0] * Nx[:, :, 1], NX[:, :, 1] * Nx[:, :, 1], Nx[:, :, 1]),
                          axis=2)
    _, s_1, Vt = np.linalg.svd(M, full_matrices=False)
    H = Vt[:, -1].reshape(-1, 3, 3)
    Hs = np.dot(np.einsum("ijk,ikm->ijm", iMSx, H), MSX[0])
    Hs /= Hs[:, 2, 2][:, None, None]

    h_X = np.empty((3, X.shape[1]))
    h_X[:2, :] = X[0].T
    h_X[2, :] = 1.0


    for i in range(len(Hs)):
        ret = optAlg(
            getError,
            Hs[i].flatten(),
            jac=getJ,
            args=(h_X, x[i]),
            method='lm'
        )
        if ret.success:
            Hs[i] = ret.x.reshape(3, 3) / ret.x[8]

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

    V = np.empty((2 * v_m_1_2.shape[1], v_m_1_2.shape[0]), dtype=np.float64)
    V[0::2] = v_m_1_2.T
    V[1::2] = (v_m_1_1 - v_m_2_2).T

    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    b = Vt[-1]
    #w = b[0]*b[2]*b[5]-b[1]*b[1]*b[5]-b[0]*b[4]*b[4]+2*b[1]*b[3]*b[4]-b[2]*b[3]*b[3]
    #d = b[0]*b[2]-b[1]*b[1]

    #alpha = np.sqrt(w/(d*b[0]))
    #beta = np.sqrt(b[0]*w/(d*d))
    #gamma = b[1]*np.sqrt(w/(b[0]*d*d))
    #uc = (b[1]*b[4]-b[2]*b[3])/d
    #vc = (b[1]*b[3]-b[0]*b[4])/d
    B = np.array([[b[0], b[1], b[2]],
                  [b[1], b[3], b[4]],
                  [b[2], b[4], b[5]]])
    B = B/B[2,2]
    try:
        KnT = np.linalg.cholesky(B)
    except LinAlgError:
        return "Did not Converge"
    K = np.linalg.inv(KnT.T)
    #K=np.array([[alpha, gamma, uc],
    #            [0, beta, vc],
    #            [0, 0, 1]])
    return K/K[2,2]
