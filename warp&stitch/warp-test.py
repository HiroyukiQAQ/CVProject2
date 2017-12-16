import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt

parameters = np.load('../Homography/homography2.npz')

H2 = parameters['H2']
print(H2)

dirname = 'warpedimg_test'

img = []


for i in range (0, 6):
    readimg = cv2.imread('../Undistortion/undistortion_results/result'+str(i+1)+'.jpg')
    img.append(readimg)
    print(i)
    print(img[i].shape)


def warpImages(img, H):
    rows, cols = img.shape[:2]

    list_of_points = np.float32([[0, 0], [0, rows], [cols, rows], [cols, 0]]).reshape(-1, 1, 2)
    list_of_points = cv2.perspectiveTransform(list_of_points, H)
    #print(list_of_points)

    x_min, y_min = np.int32(list_of_points.min(axis=0).ravel())
    # print(x_min, y_min)
    x_max, y_max = np.int32(list_of_points.max(axis=0).ravel())
    # print(x_max, y_max)

    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    output_img = cv2.warpPerspective(img, H_translation.dot(H), (x_max-x_min, y_max-y_min))

    return output_img


imgw = []

imgw.append(img[0])

cv2.imwrite(os.path.join(dirname, 'warp1.jpg'), imgw[0])

for k in range(0, 5):
    imgw.append(warpImages(img[k+1], H2[k]))
    print(imgw[k+1].shape)
    cv2.imwrite(os.path.join(dirname, 'warp' + str(k+2) + '.jpg'), imgw[k+1])

exit()