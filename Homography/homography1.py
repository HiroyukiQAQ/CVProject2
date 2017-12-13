import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt


# pics = glob.glob('../Undistortion/undistortion_results/pics1/*.jpg')
# print(pics)


surf = cv2.xfeatures2d.SURF_create()

out_fn = 'homography1.npz'



img = []
H1 = []

for i in range(0, 6):
    readimg = cv2.imread('../Undistortion/undistortion_results/result'+str(i+1)+'.jpg')
    img.append(readimg)
    print(i)
    print(img[i].shape)

for n in range(0, 5):
    # scaling_factor = 1.0
    # img2 = cv2.resize(img[n], None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    # img1 = cv2.resize(img[n+1], None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    img1 = img[n+1]
    img2 = img[n]

    kp1, des1 = surf.detectAndCompute(img1, None)
    print(len(kp1))
    kp2, des2 = surf.detectAndCompute(img2, None)
    print(len(kp2))

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for a, b in matches:
        if a.distance < 0.7 * b.distance:
            good.append(a)

    pts1 = np.float32([kp1[a.queryIdx].pt for a in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[a.trainIdx].pt for a in good]).reshape(-1, 1, 2)

    # pts1 = np.int32(pts1)
    # pts2 = np.int32(pts2)
    print(len(pts1))
    print(len(pts2))
    print(len(good))

    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    H1.append(h)
    print('H'+str(n+1)+'_'+str(n+2)+'=', H1[n])

np.savez(out_fn, H1=H1)
exit()














