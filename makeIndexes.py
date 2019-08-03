from PIL import Image
import cv2
import tensorflow as tf
import pickle, os
import datetime
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import sys

def pooldown(image,size):
	w=image.shape[0]
	h=image.shape[1]
	w_d=w//size
	h_d=h//size
	res = np.zeros([size,size,3], dtype='uint8')
	for i in range(size):
		for j in range(size):
			res[i,j,:]=image[i*w_d:(i+1)*w_d,j*h_d:(j+1)*h_d,:].mean(axis=(0,1))
	return res


def getIndex(image, size):
	pooldown(image,size)
	return res

z=0
indexing = []
inputDir = sys.argv[1].replace('/', '')
dire = inputDir+"_squared/"
files = os.listdir(dire)
files.sort()
for img in files:
	try:
		imagePath = dire+img
		base=os.path.basename(imagePath)
		imgno = int(os.path.splitext(base)[0])

		image = np.array(Image.open(imagePath))
		indexing.append([imgno,pooldown(image,5).tolist()])
		if z%500 == 0:
			print(z)
			print(datetime.datetime.now().time())
		z=z+1
	except Exception as e:
	    print(e)
	    print("Failed for "+str(img))

print(z)
with open(inputDir+"_indexes", 'w') as file:
     file.write(pickle.dumps(indexing)) 


# infile = open(filename,'rb')
# new_dict = pickle.load(infile)
# infile.close()
