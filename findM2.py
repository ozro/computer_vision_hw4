'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import cv2
import submission as sub
from helper import camera2


im1 = cv2.imread('../data/im1.png')
pts = np.load('../data/some_corresp.npz')
pts1 = pts["pts1"]
pts2 = pts["pts2"]

M = max((im1.shape[0], im1.shape[1]))
F = sub.eightpoint(pts1, pts2, M)

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
E = sub.essentialMatrix(F, K1, K2)

M1 = np.concatenate((np.eye(3), np.ones((3,1))), axis=1)
C1 = np.dot(K1, M1)
M2s = camera2(E)

for i in range(M2s.shape[2]):
    M2 = M2s[:,:,i]
    C2 = np.dot(K2, M2)
    P, err = sub.triangulate(C1, pts1, C2, pts2)
    if np.min(P[:,2]) > 0:
        break

C2 = np.dot(K2, M2)
np.savez('q3_3.npz', M2=M2, C2=C2, P=P)
