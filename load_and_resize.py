__author__ = 'chaomeng'
import cv2
import os
import numpy as np

# def fun(test, out):
#     for root, dirs, files in os.walk(test):
#         for fm in files:
#             #out.append(str(root)+'/'+str(fm))
#             out.write(str(root)+'\\'+str(fm))
#             out.write("\n")
#
# ImageList = open('imageList.txt', 'w')
# fun('jaffe', ImageList)

def resizeImage(dir):
    for root, dirs, files in os.walk(dir):
        for fm in files:
            if str(fm) == 'README':
                continue
            imgStr = str(root)+'\\'+str(fm)
            img = cv2.imread(imgStr)
            print imgStr
            smallImg = img[50:245, 56:200]
            output = 'resize\\' + str(fm)
            cv2.imwrite(output, smallImg)


resizeImage('jaffe')

# code to resize imgage
# img = cv2.imread('jaffe/KL.FE1.174.tiff')
# small = img[50:245, 56:200]
# cv2.imshow('test',small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()