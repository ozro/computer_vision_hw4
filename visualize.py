'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import submission as sub
from helper import camera2


im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png')
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


coords = np.load('../data/templeCoords.npz')
x1 = coords['x1']
y1 = coords['y1']

pts1 = np.hstack([x1, y1])
pts2 = np.zeros(pts1.shape)

for i in range(x1.shape[0]):
    x = pts1[i, 0]
    y = pts1[i, 1]
    x2,y2 = sub.epipolarCorrespondence(im1, im2, F, x, y)
    pts2[i,:] = [x2, y2]



for i in range(M2s.shape[2]):
    M2 = M2s[:,:,i]
    C2 = np.dot(K2, M2)
    P, err = sub.triangulate(C1, pts1, C2, pts2)
    if np.min(P[:,2]) > 0:
        break
np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim3d(-2.5, 1)
ax.set_ylim3d(-2.5, -0.25)
ax.set_zlim3d(8, 11.5)

ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()