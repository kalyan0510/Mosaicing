from multiprocessing import Pool
from scipy import spatial
import numpy as np
from matplotlib import pyplot as plt
import pickle
from PIL import Image
import cv2
import time
import scipy.misc
import os, sys
from numpy import linalg as LA
from matplotlib.pyplot import imshow

# python mosaicvideo.py heads vidlist 25 

mosaic_resol = int(float(sys.argv[3]))
output_block_resol = 700
blur_size = min(output_block_resol/8,6)
pooledIndex = 3 # 1-1 2-3 3-5 4-7 5-9
pooledIndexToPoolSize = [0,1,3,5,7,9]
print("Loading indexes")
infile = open(sys.argv[1]+"indexes",'rb')
indexes = pickle.load(infile)
infile.close()

index = np.array([a[0] for a in indexes])
pooln = np.array([a[pooledIndex] for a in indexes])
poolnflat = pooln.reshape(pooln.shape[0], pooln.shape[1]*pooln.shape[2]*pooln.shape[3])

print("creating tree for NearestNeighbors search")
kdtree = spatial.cKDTree(poolnflat, leafsize=pooledIndex*200)


def getAnImage(img):
	imagePath = sys.argv[1]+"/"+str(img)+".jpg"
	imagearray = np.array(Image.open(imagePath))
	return cv2.resize(imagearray, dsize=(output_block_resol, output_block_resol), interpolation=cv2.INTER_CUBIC)

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
	pooleddownavgs = []
	for image in images:
		res=pooldown(image,size)
		pooleddownavgs.append(res.reshape(size*size*3))
	diss,inds = kdtree.query(pooleddownavgs,n_jobs=-1)
	return [getAnImage(index[ind]) for ind in inds]

def bestBlockForSubstution(imgs,replacement,size):
	rep = pooldown(replacement,size).reshape(size*size*3)

	pooleddownavgs = []
	for image in imgs:
		res=pooldown(image,size)
		pooleddownavgs.append(res.reshape(size*size*3))
	min_distance = sys.maxint
	i=0
	match_index = 0
	for pooled_avg in pooleddownavgs:
		dist = LA.norm(pooled_avg - rep)
		if dist < min_distance:
			min_distance = dist
			match_index = i
		i = i+1
	return (match_index, min_distance)

def replaceEachBlock(im,width,height,poolsize, mustInclude):
	imgheight  = im.shape[0]
	imgwidth = im.shape[1]
	starttime = time.time()
	totalBlocks = (imgwidth//width)*(imgheight//height)
	#print("Total Blocks: "+str(totalBlocks))
	x=0
	min_dist = sys.maxint
	match_index = [0,0]

	newIm = np.zeros([output_block_resol*(imgheight//height),output_block_resol*(imgwidth//width),3], dtype='uint8')
	#print("\nProgress:")
	for i in range(imgwidth//width):
		X= findBestMatchParallel([im[j*height:j*height+height,i*width:i*width+width,:] for j in range(imgheight//height) ],poolsize)
		include_match_index, include_match_dist =  bestBlockForSubstution([im[j*height:j*height+height,i*width:i*width+width,:] for j in range(imgheight//height) ],mustInclude,poolsize)
		if include_match_dist < min_dist:
			min_dist = include_match_dist
			match_index[0] = include_match_index
			match_index[1] = i
		for j in range(imgheight//height):
			newIm[j*output_block_resol:(j+1)*output_block_resol,i*output_block_resol:(i+1)*output_block_resol,:] = X[j]
			x=x+1
			if x%(totalBlocks/10) == 0:
				sys.stdout.write("\r%d%%" % (x*100/totalBlocks))
				sys.stdout.flush()

	#print("\nIncluded at", match_index)
	newIm[match_index[0]*output_block_resol:(match_index[0]+1)*output_block_resol,match_index[1]*output_block_resol:(match_index[1]+1)*output_block_resol,:] = cv2.resize(mustInclude, dsize=(output_block_resol, output_block_resol), interpolation=cv2.INTER_CUBIC)
	endtime = time.time()
	#print("\t"+str(endtime-starttime)+"   "+str(x))
	posn = [match_index[0]*output_block_resol,(match_index[0]+1)*output_block_resol,match_index[1]*output_block_resol,(match_index[1]+1)*output_block_resol]
	return (newIm,posn)



def softenblocks(image, block_resol):
	w_n = image.shape[0]//block_resol
	h_n = image.shape[1]//block_resol
	height = block_resol
	width = block_resol
	bs = blur_size
	bs_k = (blur_size/4)*2+1
	for i in range(h_n):
		for j in range(w_n):
			image[(j+1)*width-bs:(j+1)*width+bs,i*height:(i+1)*height,:] = cv2.GaussianBlur(image[(j+1)*width-bs:(j+1)*width+bs,i*height:(i+1)*height,:],(bs_k,bs_k),0)
			image[j*width:(j+1)*width,(i+1)*height-bs:(i+1)*height+bs,:] = cv2.GaussianBlur(image[j*width:(j+1)*width,(i+1)*height-bs:(i+1)*height+bs,:],(bs_k,bs_k),0)
	return image


def mosaic(image, poolsize, mustInclude):
	#print("mosaicing Started")
	a = time.time()
	x,posn = replaceEachBlock(image,mosaic_resol,mosaic_resol,poolsize, mustInclude)
	x = softenblocks(x,output_block_resol)
	b = time.time()
	#print(b - a)
	return (x,posn)

def fiftyIzeImaze(image,resol= [500,500]):
	imgheight = image.shape[0]
	imgwidth = image.shape[1]
	image = square_thumb(image, imgwidth if imgheight<imgheight else imgheight)
	imgheight = image.shape[0]
	imgwidth = image.shape[1]
	xresstart = (imgheight-resol[0])/2
	xresend = imgheight - (imgheight-resol[0])/2
	yresstart = (imgwidth-resol[1])/2
	yresend = imgwidth - (imgwidth-resol[1])/2
	return image[xresstart:xresend,yresstart:yresend,:]

def changeRatio(img, wr, hr):
	width, height = img.size
	if width/height > wr/hr:
		upper = 0
		lower = height
		left = width/2 - (wr*height/hr)/2
		right = width/2 + (wr*height/hr)/2
	else :
		left = 0
		right = width
		upper = height/2 - (width*hr/wr)/2
		lower = height/2 + (width*hr/wr)/2
	img = img.crop((left, upper, right, lower))
	return np.array(img)


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def videoMaker(targetImgsfolder, video, framerate, video_resol, zoomoutTime):
	width = video_resol[0]
	height = video_resol[1] 
	max_resol = (max(width, height),max(width, height))
	#print(float(width)/height)
	print("Video Mosaicing started")
	if float(width)/height <= 1:
	  	ub = 0
	  	lwb = height
	  	lb = height/2 - (width)/2
	  	rb = height/2 + (width)/2
	else :
	  	lb = 0
	  	rb = width
	  	ub = width/2 - (height)/2
	  	lwb = width/2 + (height)/2

	#print("borders: ",[ub,lwb,lb,rb])

	framerate = int(framerate)
	images = []
	for filename in os.listdir(targetImgsfolder):
		try:
			if(filename.endswith(".jpg")):
				img = Image.fromarray(cv2.imread(targetImgsfolder+"/"+filename)[:,:,::-1]).convert('RGB')
				images.append(image_resize(np.array(img),width =1000 ))
				#img.show()
		except Exception as e:
			print(e)
	images.append(images[0])
	print("Total #images: "+str(len(images)))
	for i in range(len(images)):
		images[i] = np.array(changeRatio(Image.fromarray(images[i]), 10, 10))

	for i in xrange(len(images)-1):
		print("Processing image no. : "+str(i+1))
		current = images[i]
		target = images[i+1]

		mosaic_img, posn = mosaic(target,pooledIndexToPoolSize[pooledIndex], mustInclude = current)
		resized_current_img = cv2.resize(current[:,:,::-1], dsize=max_resol, interpolation=cv2.INTER_CUBIC)
		resized_current_img = resized_current_img[ub:lwb,lb:rb,:]
		print("\nProgress: ")
		for j in range(framerate):
			video.write(resized_current_img)

		zoomoutframes = zoomoutTime * framerate
		print("Total Frames: "+str(zoomoutframes))
		left = posn[2]
		up = posn[0]
		right = posn[3]
		down = posn[1]
		height = mosaic_img.shape[0]
		width = mosaic_img.shape[1]
		#print height, width
		t_h = target.shape[0]
		t_w = target.shape[1]
		mosaic_img[up:down,left:right,:] = cv2.resize(current, dsize=(down-up, right-left), interpolation=cv2.INTER_CUBIC)
		for f in range(zoomoutframes):
			j = (f*f*f*f*1.0)/(zoomoutframes*zoomoutframes*zoomoutframes)
			l = int(left - (left*j)/zoomoutframes)
			u = int(up - (up*j)/zoomoutframes)
			r = int(right + ((width-right)*j)/zoomoutframes)
			d = int(down + ((height-down)*j)/zoomoutframes)
			if j < int(zoomoutframes*0.8):
				frm = cv2.resize((mosaic_img[u:d,l:r,:])[:,:,::-1], dsize=max_resol, interpolation=cv2.INTER_LINEAR)
				frm = frm[ub:lwb,lb:rb,:]
				video.write(frm)
			else:
				a = cv2.resize((mosaic_img[u:d,l:r,:]), dsize=max_resol, interpolation=cv2.INTER_LINEAR)
				b = cv2.resize((target[(u*t_h)/height:(d*t_h)/height,(l*t_w)/height:(r*t_w)/height,:] ), dsize=max_resol, interpolation=cv2.INTER_LINEAR)
				weight = (j - zoomoutframes*0.8)/(zoomoutframes*0.2);
				frm = np.uint8(np.add((1-weight)*a, weight*b))[ub:lwb,lb:rb,::-1]
				video.write(frm)
			#if f%100 == 0:
			sys.stdout.write("\r%s" % str(f)+" frames completed out of "+str(zoomoutframes))
			sys.stdout.flush()

		print("\n")


targetImgsfolder = sys.argv[2]

framerate = 15.0
zoomoutTime = 5;
video_resol = (1100,750)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

video = cv2.VideoWriter(str(targetImgsfolder)+".avi", fourcc, framerate, video_resol)
videoMaker(targetImgsfolder, video, framerate,video_resol, zoomoutTime)
cv2.destroyAllWindows()

video.release()
  