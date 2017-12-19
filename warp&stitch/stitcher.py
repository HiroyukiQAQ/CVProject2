import numpy as np
import cv2

parameters = np.load('../Homography/homography2.npz')

H2 = parameters['H2']

imgu = []

for i in range(0, 6):
    readimg = cv2.imread('../Undistortion/undistortion_results/result'+str(i+1)+'.jpg')
    imgu.append(readimg)
    # print(i)

imgw = []

imgw.append(imgu[0])

for i in range(1, 6):
    rows1, cols1 = imgu[0].shape[:2]
    rows2, cols2 = imgu[i].shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    # print(list_of_points_1.shape)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    # print(temp_points)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H2[i-1])
    # print(list_of_points_2.shape)
    # print(list_of_points_2)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    img = cv2.warpPerspective(imgu[i], H2[i-1], (x_max-x_min, y_max-y_min))
    print(img.shape)
    imgw.append(img)


# rows, cols = imgw[5].shape[:2]
# print(rows, cols)

tx = []
ty = []
for i in range(1, 6):
    readimg = cv2.imread('warpedimg/warp'+str(i+1)+'.jpg')
    h1, w1 = readimg.shape[:2]
    h2, w2 = imgw[i].shape[:2]
    tx.append(w2-w1)
    ty.append(h2-h1)
    # print('tx: ', tx[i-1])
    # print('ty: ', ty[i-1])


# result_img = np.zeros((rows, cols, 3))


def stitch_img(l, r, a):
    rowsl, colsl = l.shape[:2]
    rowsr, colsr = r.shape[:2]

    lp = np.float32([[0, 0], [0, rowsl], [colsl, rowsl], [colsl, 0]]).reshape(-1, 1, 2)
    rp = np.float32([[0, 0], [0, rowsr], [colsr, rowsr], [colsr, 0]]).reshape(-1, 1, 2)


    xl_min, yl_min = np.int32(lp.min(axis=0).ravel())
    print(xl_min, yl_min)
    xl_max, yl_max = np.int32(lp.max(axis=0).ravel())
    print(xl_max, yl_max)

    xr_min, yr_min = np.int32(rp.min(axis=0).ravel())
    print(xr_min, yr_min)
    xr_max, yr_max = np.int32(rp.max(axis=0).ravel())
    print(xr_max, yr_max)

    # x_min = min(xl_min, xr_min)
    x_max = max(xl_max, xr_max)
    # y_min = min(yl_min, yr_min)
    y_max = max(yl_max, yr_max)

    super_img = np.zeros((y_max, x_max, 3))

    for x in range(0, tx[a]):
        for y in range(yl_min, yl_max):
            super_img[y][x] = l[y][x]

    for x in range(tx[a]+1, xl_max):
        for y in range(0, min(yl_max, yr_max)):
            super_img[y][x] = r[y][x]

    for x in range(xl_max+1, xr_max):
        for y in range(yr_min, yr_max):
            super_img[y][x] = r[y][x]

    return super_img


result = imgw[0]

for j in range(0, 5):
    print(j)
    left = result
    right = imgw[j+1]
    stitch_result = stitch_img(left, right, j)
    print('stitch.size=', stitch_result.shape)
    result = stitch_result

cv2.imwrite('stitichimg/superimage3.jpg', result)

exit()