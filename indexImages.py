from PIL import Image
import cv2
import tensorflow as tf
import pickle
import datetime
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import sys,os

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
i=0;
dire = sys.argv[1]
for imgFile in os.listdir(dire):
	
	try:
		imagePath = dire+"/"+imgFile
		#print imagePath, int(imgFile.split(".")[0])
		image = np.array(Image.open(imagePath))
		indexing.append([int(imgFile.split(".")[0]),pooldown(image,1).tolist(),pooldown(image,3).tolist(),pooldown(image,5).tolist(),pooldown(image,7).tolist(),pooldown(image,9).tolist()])
		if i%500 == 0:
			print(i)
			print(datetime.datetime.now().time())
		i=i+1
	except Exception as e:
	    print(e)
	    print("Failed for "+str(i))

print(i)
with open(dire+"indexes", 'w') as file:
     file.write(pickle.dumps(indexing)) 


# infile = open(filename,'rb')
# new_dict = pickle.load(infile)
# infile.close()
