__author__ = 'chaomeng'
import cv2
import numpy as np

img = cv2.imread('jaffe/KA.HA2.30.tiff')
img = img[50:240, 56:200]
fast = cv2.FastFeatureDetector(50)
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))
cv2.imwrite('fast_small.png',img2)