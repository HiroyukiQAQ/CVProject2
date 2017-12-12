import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt

parameters = np.load('../Homography/homography2.npz')

H2 = parameters['H2']
print(H2)

dirname = 'warpedimg'

img = []


for i in range (0, 6):
    readimg = cv2.imread('../Undistortion/undistortion_results/result'+str(i+1)+'.jpg')
    img.append(readimg)
    print(i)
    print(img[i].shape)


def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    print(list_of_points_1.shape)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    print(list_of_points_2.shape)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    # [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    # [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    [x_min, y_min] = np.int32(list_of_points_2.min(axis=0).ravel() - 0.5)
    print(x_min, y_min)
    [x_max, y_max] = np.int32(list_of_points_2.max(axis=0).ravel() + 0.5)
    print(x_max, y_max)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    #output_img = cv2.warpPerspective(img2, H_translation.dot(H), (1548, 1152))
    #output_img = cv2.warpPerspective(img2, H, (x_max - x_min, y_max - y_min))
    # output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    return output_img


imgw = []

imgw.append(img[0])

cv2.imwrite(os.path.join(dirname, 'warp1.jpg'), imgw[0])

for k in range(0, 5):
    imgw.append(warpImages(img[0], img[k+1], H2[k]))
    # print(imgw[k+1].shape)
    # imgw.append(cv2.warpPerspective(img[k+1], H2[k], (1548, 1152)))
    cv2.imwrite(os.path.join(dirname, 'warp' + str(k+2) + '.jpg'), imgw[k+1])

exit()