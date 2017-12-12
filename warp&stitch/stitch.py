import numpy as np
import cv2

stitcher = cv2.createStitcher(False)

img = []

for i in range (0, 6):
    readimg = cv2.imread('warpedimg/warp'+str(i+1)+'.jpg')
    img.append(readimg)
    print(i)
    print(img[i].shape)

# for j in range (0,4):
#     (tuple, result) = stitcher.stitch((img[0], img[j+1]))
#     # print(j)
#     print(tuple)
#     img[0] = result
(tuple, result) = stitcher.stitch((img[0], img[1], img[2], img[3], img[4], img[5]))
print(tuple)

cv2.imwrite('stitichimg/superimage.jpg', result)

exit()

