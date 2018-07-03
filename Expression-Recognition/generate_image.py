import os
import csv
#import cv2.cv as cv
#import cv2
import numpy as np
from PIL import Image
import scipy.misc

import matplotlib.pyplot as plt

csvfile1 = open('./fer2013/fer2013.csv','r')
reader1 = csv.reader(csvfile1)
#pixel = [row[1] for row in reader]
i = 0
emotion = []
usage = []
for item in reader1: 
	if reader1.line_num == 1:  
		continue
	#print len(item)
	#print item[0][0]
	emotion.extend(item[0])
	#pixels.extend(item[1].split(' '))
	usage.extend(item[2])
	#print len(pixels)
	#pixel = pixel.reshape(48,48,1)
	#img = cv2.imread('8-381.jpg')
	#img = img.resize(48,48,1)
	#im.itemset(item)
	#plt.imshow(img)
	#image = Image.fromarray(item)
	#print(item.shape)
	#print(img.size)
	#cv2.imwrite("./cat2.jpg", item, [int(cv2.IMWRITE_JPEG_QUALITY), 5])  
	#cv2.imwrite('./train_data_image/{}'.format(str(i)+'.jpg',item))
	i += 1
#print emotion[0],usage[0]
i = 0
pixel = []
img = []
csvfile2 = open('./fer.csv','r')
reader2 = csv.reader(csvfile2)
for item in reader2:
	print len(item)
	item = map(int,item)
	pixel = np.array(item).reshape(48,48)
	print pixel
	scipy.misc.imsave('./trian_data/{}'.format(str(i)+'-'+str(emotion[i])+'.jpg'),pixel)
	#cv.SaveImage('./dataset/'+str(emotion[i])+'-'+str(i),fromarray(pixel))
	#cv2.imwrite('./dataset/'+str(emotion[i])+'-'+str(i),pixel)
	i += 1
	#pixel = pixel.append(item)
#print pixel
#img = np.array(pixel).reshape(-1,48,48)

'''
for i in range(35887):
	cv2.imwrite('./dataset/'+str(emotion[i])+'-'+str(i)+'-'+usage[i],img[i])
'''
