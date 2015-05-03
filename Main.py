__author__ = 'chaomeng'

import csv
import io
import cv2

trainImage = []
testImage = []
imageList = open('imageList.txt', 'r')
partList = open('partOfList.txt', 'w')
for line in imageList:
    part = line.split('.')
    img = cv2.imread(line)
    if '1' in part[1]:
        trainImage.append(img)
        partList.write(line.replace("jaffe", "resize"))
        #partList.write('\n')
    else:
        testImage.append(img)

