from multiprocessing import Pool
from scipy import spatial
import numpy as np
from matplotlib import pyplot as plt
import pickle
from PIL import Image
import cv2
import time
import scipy.misc
import sys

# python offlineMosaicGen.py foldername indexname 25(size of each block in final img) imputimage.jpg

mosaic_resol = int(float(sys.argv[3]))
output_block_resol = 50
blur_size = 3
pooledIndex = 3 # 1-1 2-3 3-5 4-7 5-9



pooledIndexToPoolSize = [0,1,3,5,7,9]
print("Loading indexes")
infile = open(sys.argv[2],'rb')
indexes = pickle.load(infile)
infile.close()

index = np.array([a[0] for a in indexes])
pooln = np.array([a[pooledIndex] for a in indexes])
poolnflat = pooln.reshape(pooln.shape[0], pooln.shape[1]*pooln.shape[2]*pooln.shape[3])

print("creating tree for NearestNeighbors search")
kdtree = spatial.cKDTree(poolnflat, leafsize=pooledIndex*200)


print("Loading Images....")
# allimages = dict()
# for i in index:
# 	imagePath = sys.argv[1]+"/"+str(i)+".jpg"
# 	imagearray = np.array(Image.open(imagePath))
# 	# unnecessary if both resols are same 
# 	if output_block_resol == 51:
# 		allimages[i] =imagearray
# 	else:
# 		allimages[i] = cv2.resize(imagearray, dsize=(output_block_resol, output_block_resol), interpolation=cv2.INTER_CUBIC)

def getAnImage(img):
	imagePath = sys.argv[1]+"/"+str(img)+".jpg"
	imagearray = np.array(Image.open(imagePath))
	return cv2.resize(imagearray, dsize=(output_block_resol, output_block_resol), interpolation=cv2.INTER_CUBIC)


print("Completed loading")

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


# def findBestMatch(image,size):
# 	res = np.zeros([size,size,3], dtype='uint8')
# 	for i in range(3):
# 		res[:,:,i]=cv2.resize(image[:,:,i], dsize=(size,size))
# 	#starttime = time.time()
# 	dis,ind = kdtree.query(res.reshape(size*size*3))
# 	#endtime = time.time()
	
# 	return allimages[index[ind]]

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
	print(index[inds])
	return [getAnImage(index[ind]) for ind in inds]

#changing func to maintain own image before returning
def replaceEachBlock(im,width,height,poolsize):
	imgheight  = im.shape[0]
	imgwidth = im.shape[1]
	starttime = time.time()
	totalBlocks = (imgwidth//width)*(imgheight//height)
	print("Total Blocks: "+str(totalBlocks))
	x=0
	newIm = np.zeros([output_block_resol*(imgheight//height),output_block_resol*(imgwidth//width),3], dtype='uint8')
	for i in range(imgwidth//width):
		X= findBestMatchParallel([im[j*height:j*height+height,i*width:i*width+width,:] for j in range(imgheight//height) ],poolsize)
		for j in range(imgheight//height):
			newIm[j*output_block_resol:(j+1)*output_block_resol,i*output_block_resol:(i+1)*output_block_resol,:] = X[j]
			x=x+1
			if x%(totalBlocks/10) == 0:
				print(x*100.0/totalBlocks)
	endtime = time.time()
	print("\t"+str(endtime-starttime)+"   "+str(x))
	return newIm



def softenblocks(image, block_resol):
	w_n = image.shape[0]//block_resol
	h_n = image.shape[1]//block_resol
	height = block_resol
	width = block_resol
	bs = blur_size
	bs_k = (blur_size/4)*2+1
	for i in range(h_n):
		for j in range(w_n):
			#print((j+1)*width-bs,(j+1)*width+bs,i*height,(i+1)*height)
			image[(j+1)*width-bs:(j+1)*width+bs,i*height:(i+1)*height,:] = cv2.GaussianBlur(image[(j+1)*width-bs:(j+1)*width+bs,i*height:(i+1)*height,:],(bs_k,bs_k),0)
			image[j*width:(j+1)*width,(i+1)*height-bs:(i+1)*height+bs,:] = cv2.GaussianBlur(image[j*width:(j+1)*width,(i+1)*height-bs:(i+1)*height+bs,:],(bs_k,bs_k),0)
	return image


def mosaic(image, poolsize):
	print("mosaicing Started")
	a = time.time()
	x = replaceEachBlock(image,mosaic_resol,mosaic_resol,poolsize)
	x = softenblocks(x,output_block_resol)
	b = time.time()
	print(b - a)
	return x

def presetImaze(image,resol= [500,500]):
	imgheight = image.shape[0]
	imgwidth = image.shape[1]
	xresstart = (imgheight-resol[0])/2
	xresend = imgheight - (imgheight-resol[0])/2
	yresstart = (imgwidth-resol[1])/2
	yresend = imgwidth - (imgwidth-resol[1])/2
	return image[xresstart:xresend,yresstart:yresend,:]

images = sys.argv[4:]
for image in images:
	print("mosaicing "+image)
	imagePath = str(image)
	imagearray = np.array(Image.open(imagePath))
	imagearray=imagearray[:,:,:3]
	print(imagearray.shape)
	imagearray = presetImaze(imagearray,[(imagearray.shape[0]/50)*50,(imagearray.shape[1]/50)*50])
	imagearray = mosaic(imagearray,pooledIndexToPoolSize[pooledIndex])
	img = Image.fromarray(imagearray, 'RGB')
	img.show()
	img.save(str(image)+"_mosaic.jpg")



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

  