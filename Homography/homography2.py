import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt

parameters = np.load('homography1.npz')

H1 = parameters['H1']

out_fn = 'homography2.npz'

H2=[]

H2.append(H1[0])  #H21  0

H2.append(np.dot(H2[0], H1[1]))  #H31  1

H2.append(np.dot(H2[1], H1[2]))  #H41  2

H2.append(np.dot(H2[2], H1[3]))  #H51  3

H2.append(np.dot(H2[3], H1[4]))  #H61  4


for i in range(0, 5):
    H2[i] = (H2[i])/[H2[i][2,2]]
    print('H1_'+str(i+2)+'=', H2[i])

np.savez(out_fn, H2=H2)

exit()
