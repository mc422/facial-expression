__author__ = 'chaomeng'
import cv2
import numpy as np

img = cv2.imread('jaffe/KA.FE1.45.tiff')
orb = cv2.ORB(180)
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
print len(kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
cv2.imwrite('orb_test.png',img2)

