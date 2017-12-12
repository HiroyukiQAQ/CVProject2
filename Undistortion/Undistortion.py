import numpy as np
import cv2
import glob
import os

dirname = 'undistortion_results'

parameters = np.load('../CameraCalibration/parameters.npz')
mtx = parameters['mtx']
dist = parameters['dist']

print(mtx)

img = []

for i in range(0, 6):
    readimg = cv2.imread('../pics/'+str(i+1)+'.jpg')
    img.append(readimg)
    print(i)
    print(img[i].shape)


for j in range(0, 6):
    print(j)
    h, w = img[j].shape[:2]
    img_size = img[j].shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1)
    dst = cv2.undistort(img[j], mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(os.path.join(dirname, 'result' + str(j+1))+'.jpg', dst)

exit()