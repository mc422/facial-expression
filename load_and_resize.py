__author__ = 'chaomeng'
import cv2
import os
import csv
import io
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

def gather_expression(line, file, expression):
    part = line.split('.')
    if expression in part[1]:
        file.write(line.replace("jaffe", "resize"))

def gather_image_index(line, file, index):
    part = line.split('.')
    if str(index) in part[1]:
        #trainImage.append(img)
        file.write(line.replace("jaffe", "resize"))
    #else:
        #testImage.append(img)

# resize images
resizeImage('jaffe')

# create image list file
trainImage = []
testImage = []
imageList = open('imageList.txt', 'r')
partList = open('partOfList.txt', 'w')
angryList = open('ANGRY.txt', 'w')
happyList = open('HAPPY.txt', 'w')
dispoint = open('DISP.txt', 'w')
expression = ['AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU']
for line in imageList:
    gather_image_index(line, partList, 1)
    gather_expression(line, angryList, expression[0])
    gather_expression(line, dispoint, expression[1])
    gather_expression(line, happyList, expression[3])


