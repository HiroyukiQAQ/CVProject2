import numpy as np
import cv2
from numpy.linalg import inv

img = cv2.imread('../warp&stitch/stitichimg/superimage.jpg')

parameters = np.load('../CameraCalibration/parameters.npz')
K = parameters['mtx']

theta = np.radians(45)   # clockwise 45 degree
sin = np.sin(theta)
cos = np.cos(theta)

I = np.eye(3, dtype=int)

n = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])  # cross product matrix of y axis
# since i'm using the landscape mode. the gravity axis is actually y axis

dot = np.dot(n, n)

R = I - (n*sin) + (dot*(1 - cos))  # Rodrigues' Rotation Matrix
print(R)

H = np.dot(np.dot(K, R), inv(K))  # H = KR(K^-1)
print(H)

(h, w) = img.shape[:2]

rotate_img = cv2.warpPerspective(img, H, (w, h))
print(rotate_img.shape)

cv2.imwrite('rotate_superimage.jpg', rotate_img)
exit()