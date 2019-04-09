"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import cv2

def testEightPoint():
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')
    pts = np.load('../data/some_corresp.npz')
    pts1 = pts["pts1"]
    pts2 = pts["pts2"]
    M = max((im1.shape[0], im1.shape[1]))
    F = eightpoint(pts1, pts2, M)
    from helper import displayEpipolarF
    displayEpipolarF(im1, im2, F)
    np.savez('q2_1.npz', F=F, M=M)

def testSevenPoint():
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')
    pts = np.load('../data/some_corresp.npz')
    indexes = [0, 1, 17, 19, 58, 91, 109]
    # 157 231
    # 309 284
    # 132 329
    # 474 384
    # 62 154
    # 425 223
    # 236 159
    pts1 = pts["pts1"][indexes]
    pts2 = pts["pts2"][indexes]
    M = max((im1.shape[0], im1.shape[1]))
    Farray = sevenpoint(pts1, pts2, M)
    
    F = Farray[0]
    print(F)
    from helper import displayEpipolarF
    displayEpipolarF(im1, im2, F)
    np.savez('q2_2.npz', F=F, M=M, pts1=pts1, pts2=pts2)

def testEssentialMatrix():
    im1 = cv2.imread('../data/im1.png')
    pts = np.load('../data/some_corresp.npz')
    pts1 = pts["pts1"]
    pts2 = pts["pts2"]

    M = max((im1.shape[0], im1.shape[1]))
    F = eightpoint(pts1, pts2, M)
    print(F)

    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    print(E)

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    pts1 = pts1/M
    pts2 = pts2/M
    u = pts1[:, 0]
    u1 = pts2[:, 0]
    v = pts1[:, 1]
    v1 = pts2[:, 1]

    # Construction A matrix
    A = np.vstack([u*u1, v*u1, u1, u*v1, v*v1, v1, u, v, np.ones(u.shape)])
    # Get eigenvector
    _,_,V = np.linalg.svd(A.T)
    F = np.reshape(V[-1, :], (3,3))

    # Local minimization
    from helper import refineF
    F = refineF(F, pts1, pts2) 

    # Unscale F using T representing scaling by 1/M
    T = np.array([[1/M,   0,   0],
                  [  0, 1/M,   0],
                  [  0,   0,   1]])

    unscaledF = np.dot(np.dot(np.transpose(T), F), T)
    return unscaledF

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    pts1 = pts1/M
    pts2 = pts2/M
    u = pts1[:, 0]
    u1 = pts2[:, 0]
    v = pts1[:, 1]
    v1 = pts2[:, 1]

    # Construction A matrix
    A = np.vstack([u*u1, v*u1, u1, u*v1, v*v1, v1, u, v, np.ones(u.shape)])

    # Get eigenvector
    _,_,V = np.linalg.svd(A.T)
    F1 = np.reshape(V[-1, :], (3,3))
    F2 = np.reshape(V[-2, :], (3,3))

    # Solve polynomial
    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
    a0 = fun(0)
    a1 = 2 * (fun(1) - fun(-1)) / 3 - (fun(2) - fun(-2)) / 12
    a2 = 0.5 * fun(1) + 0.5 * fun(-1) - a0
    a3 = fun(1) - a0 - a1 - a2
    alphas = np.roots(np.array([a3, a2, a1, a0]))    

    # Refine
    T = np.array([[1/M,   0,   0],
                  [  0, 1/M,   0],
                  [  0,   0,   1]])
    Farray = []
    for alpha in alphas:
        F = alpha*F1 + (1-alpha)*F2

        from helper import refineF
        F = refineF(F, pts1, pts2)
        F = np.dot(np.dot(T.T, F), T)
        Farray.append(F)

    return Farray

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = np.dot(np.dot(K2.T, F), K1)
    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]

    num_pts = x1.shape[0]
    W = np.zeros((num_pts,4))
    for i in range(num_pts):
        c11 = C1[0,:]
        c12 = C1[1,:]
        c13 = C1[2,:]
        c21 = C2[0,:]
        c22 = C2[1,:]
        c23 = C2[2,:]
        xi1 = x1[i]
        yi1 = y1[i]
        xi2 = x2[i]
        yi2 = y2[i]

        A = np.vstack([c11 - c13 * xi1, 
                       c12 - c13 * yi1,
                       c21 - c23 * xi2,
                       c22 - c23 * yi2])
        _,_,V = np.linalg.svd(A)
        w = V[-1, :]
        # Normalize W (Nx3)
        W[i, :] = w/w[3]

    # Reproject W (Nx3) into W_r (3xN)
    W_r1 = np.dot(C1, W.T)
    W_r2 = np.dot(C2, W.T)

    # Normalize the projection
    W_r1 = (W_r1[:2] / W_r1[2]).T
    W_r2 = (W_r2[:2] / W_r2[2]).T

    err = 0
    for i in range(pts1.shape[1]):
        err += np.linalg.norm(W_r1[i,:]- pts1[i,:]) + np.linalg.norm(W_r2[i,:]- pts2[i,:])

    P = W[:, 0:3]
    return P, err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass

if __name__ == "__main__":
    # testEightPoint()
    # testSevenPoint()
    # testEssentialMatrix()
    pass