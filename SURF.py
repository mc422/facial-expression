__author__ = 'chaomeng'
import cv2
import numpy as np

img = cv2.imread('jaffe/KA.HA2.30.tiff')
surf = cv2.SURF(8000)
kp, des = surf.detectAndCompute(img,None)
print des.shape
print surf.descriptorSize()
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
cv2.imshow('test',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()