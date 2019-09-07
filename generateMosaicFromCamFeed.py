from multiprocessing import Pool
from scipy import spatial
import numpy as np
from matplotlib import pyplot as plt
import pickle
from PIL import Image
import cv2
import time
import sys
import time
from datetime import datetime

dire = sys.argv[1]
mosaic_resol = 25
op_block_res = 80
blur_size = 10
pooledIndex = 3
   # 1-1 2-3 3-5 4-7
pooledIndexToPoolSize = [0,1,3,5,7]
print("Loading indexes....")
infile = open(dire+'indexes','rb')
indexes = pickle.load(infile)
infile.close()

index = np.array([a[0] for a in indexes])
pooln = np.array([a[pooledIndex] for a in indexes])
poolnflat = pooln.reshape(pooln.shape[0], pooln.shape[1]*pooln.shape[2]*pooln.shape[3])

print("creating tree for K-D Tree search...")
kdtree = spatial.cKDTree(poolnflat, leafsize=pooledIndex*400)





#color_grid_size = 64
#neighbour = [[[[[[] for k in xrange(color_grid_size)] for j in xrange(color_grid_size)] for i in xrange(color_grid_size)]  for x in xrange(grid_size)] for i in xrange(grid_size)]

print("Memoising the nearest neighbours")
print("Loading Images...."+str(index.shape)+"  -  "+str(np.max(index+1)))
starttime = time.time()
allimages = [0 for k in xrange(np.max(index+1))]
for i in index:
	imagePath = dire+"/"+str(i)+".jpg"
	imagearray = np.array(Image.open(imagePath))
	allimages[i] = cv2.resize(imagearray, dsize=(op_block_res, op_block_res), interpolation=cv2.INTER_CUBIC)

endtime = time.time()
print("time for loading images to memory")
print(endtime-starttime)

# starttime = time.time()
# for i in index:
# 	x=allimages[i]
# endtime = time.time()
# print("time for loading images from memory")
# print(endtime-starttime)

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



def findBestMatchParallel(images,size):
	#print("size of images"+str(len(images)))
	pooleddownavgs = []
	for image in images:
		res=pooldown(image,size)
		pooleddownavgs.append(res.reshape(size*size*3))
	#starttime = time.time()
	diss,inds = kdtree.query(pooleddownavgs,n_jobs=-1)
	#endtime = time.time()
	#print(endtime-starttime)
	#print(inds)
	return [allimages[index[ind]] for ind in inds]

#changing func to maintain own image before returning
def replaceEachBlock(im,width,height,poolsize):
	imgheight  = im.shape[0]
	imgwidth = im.shape[1]
	totalBlocks = (imgwidth//width)*(imgheight//height)
	newIm = np.zeros([op_block_res*(imgheight//height),op_block_res*(imgwidth//width),3], dtype='uint8')
	for i in range(imgwidth//width):
		X= findBestMatchParallel([im[j*height:j*height+height,i*width:i*width+width,:] for j in range(imgheight//height) ],poolsize)
		for j in range(imgheight//height):
			newIm[j*op_block_res:(j+1)*op_block_res,i*op_block_res:(i+1)*op_block_res,:] = X[j]

	return newIm



def softenblocks(image, block_resol):
	w_n = image.shape[0]//block_resol
	h_n = image.shape[1]//block_resol
	height = block_resol
	width = block_resol
	bs = blur_size
	for i in range(h_n):
		for j in range(w_n):
			image[(j+1)*width-bs:(j+1)*width+bs,i*height:(i+1)*height,:] = cv2.GaussianBlur(image[(j+1)*width-bs:(j+1)*width+bs,i*height:(i+1)*height,:],(bs/2,bs/2),0)
			image[j*width:(j+1)*width,(i+1)*height-bs:(i+1)*height+bs,:] = cv2.GaussianBlur(image[j*width:(j+1)*width,(i+1)*height-bs:(i+1)*height+bs,:],(bs/2,bs/2),0)
	return image

def mosaic(image, poolsize):
	#a = time.time()
	x=replaceEachBlock(image,mosaic_resol,mosaic_resol,poolsize)
	x=softenblocks(x, op_block_res)
	#b = time.time()
	#print(b - a)
	return x
	




def show_webcam(mirror=False):
	cam = cv2.VideoCapture(0)
	cv2.namedWindow('mosaic',cv2.WINDOW_NORMAL)
	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	resol = [500,600]
	xresstart = (720-resol[0])/2
	xresend = 720 - (720-resol[0])/2
	yresstart = (1280-resol[1])/2
	yresend = 1280 - (1280-resol[1])/2

	while True:
	#for i in range(1,100):
		now = datetime.now()
		ret_val, img = cam.read()
		
		img = cv2.flip(img, 1)
		cv2.imshow('image', cv2.resize(img, (240, 130)))
		img=cv2.resize(img, dsize=(640, 360))

		#img = img[xresstart:xresend,yresstart:yresend,:]
		
		#print(img.shape)
		#mosaic(img,1)
		cv2.imshow('mosaic', cv2.resize(mosaic(img,pooledIndexToPoolSize[pooledIndex]), dsize=(960, 540)))
		later = datetime.now()
		difference = (later - now).total_seconds()
		print(1/difference)
		
		#cv2.imshow('image', img)
		if cv2.waitKey(1) == 27: 
			break  # esc to quit
	cv2.destroyAllWindows()



def main():					
	show_webcam(mirror=True)
	


if __name__ == '__main__':
	main()




# for i in range(1,10):
# 	imagePath = "thumbs/"+str(i)+".jpg"
# 	imagearray = np.array(Image.open(imagePath))
# 	plt.imshow(imagearray)
# 	plt.show()
# 	starttime = time.time()
# 	res=findBestMatch(imagearray,1)
# 	end = time.time()
# 	print(end-starttime)
# 	plt.imshow(res)
# 	plt.show()

  