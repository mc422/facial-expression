__author__ = 'chaomeng'
import cv2
import numpy as np

img = cv2.imread('jaffe/KA.FE1.45.tiff')
orb = cv2.ORB(180)
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
kp1 = kp[0:50]
kp2 = kp[50:len(kp)-1]
color = (0,255,0)
# draw only keypoints location,not size and orientation
#img = cv2.drawKeypoints(img,kp1,color=(0,255,0), flags=0)
pic1 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
pic2 = cv2.drawKeypoints(img,kp1,color=(255,0,0), flags=0)
pic3 = cv2.drawKeypoints(pic2,kp2,color=(0,0,255), flags=0)

cv2.imshow('test1',pic1)
cv2.imshow('test2',pic2)
cv2.imshow('test3',pic3)
cv2.waitKey(0)
cv2.destroyAllWindows()

