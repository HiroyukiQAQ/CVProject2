import numpy as np
import cv2
import glob
import os

######  Input  #######

images = glob.glob('pics/*.jpg')

#######  Output  #########

out_fn = 'parameters.npz'

#######  Code  ########

dirname = 'pics_with_lines'
# os.mkdir(dirname)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*5, 3), np.float32)
objp[:, :2] = np.mgrid[0:5, 0:7].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# print(images)
i = 0
img_size = []
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5, 7), None)
    # If found, add object points, image points (after refining them)
    if ret is True:
        # print(fname)
        i = i + 1
        print(i)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        img_size = gray.shape[::-1]
        # # Draw and display the corners
        cv2.drawChessboardCorners(img, (5, 7), corners2, ret)
        # cv2.imwrite(os.path.join(dirname, 'pic' + str(i) + '.jpg'), img)
        # cv2.imshow('img', img)
        # cv2.waitKey(2000)

rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# mtx = np.matrix('1358 0 800; 0 1358 600; 0 0 1')
print(rms)
print(mtx)
print(dist)


np.savez(out_fn, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
exit()