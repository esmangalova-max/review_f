# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 12:23:52 2025

@author: User
"""


import cv2
import numpy as np

cap = cv2.VideoCapture('rtsp://192.168.32.166:554/av0_0')

it = 0

ret, frame = cap.read()

#cv2.imshow('frame', frame)
#cv2.waitKey(0)


#cap.release()
#cv2.destroyAllWindows()

fr = np.zeros(frame.shape)

for i in range(frame.shape[0]):
    for j in range(frame.shape[1]):
        for k in range(frame.shape[2]):
            fr[i, j, k] = int(frame[i, j, k])
