from multiprocessing import Pool
from scipy import spatial
import numpy as np
from matplotlib import pyplot as plt
import pickle
from PIL import Image
import cv2
import time

mosaic_resol = 30


print("Loading indexes")
infile = open('multipleindexes','rb')
indexes = pickle.load(infile)
infile.close()

index = np.array([a[0] for a in indexes])
pool1 = np.array([a[1] for a in indexes])
pool1flat = pool1.reshape(pool1.shape[0], pool1.shape[1]*pool1.shape[2]*pool1.shape[3])

print("creating tree for NearestNeighbors search")
kdtree = spatial.KDTree(pool1flat, leafsize=150)


print("Loading Images....")
allimages = dict()
for i in index:
	imagePath = "thumbs/"+str(i)+".jpg"
	imagearray = np.array(Image.open(imagePath))
	allimages[i] = cv2.resize(imagearray, dsize=(mosaic_resol, mosaic_resol), interpolation=cv2.INTER_CUBIC)


print("Completed loading")



def findBestMatch(image,size):
	res = np.zeros([size,size,3], dtype='uint8')
	for i in range(3):
		res[:,:,i]=cv2.resize(image[:,:,i], dsize=(size,size))
	#starttime = time.time()
	dis,ind = kdtree.query(res.reshape(3))
	#endtime = time.time()
	
	return allimages[index[ind]]

def findBestMatchIterativesizeone(imagelist):
	ans = []
	for image in imagelist:
		res = np.zeros([1,1,3], dtype='uint8')
		for i in range(3):
			res[:,:,i]=cv2.resize(image[:,:,i], dsize=(1,1))
		dis,ind = kdtree.query(res.reshape(3))
		ans.append(allimages[index[ind]])
	return anss



def replaceEachBlock(im,height,width):
	imgwidth  = im.shape[0]
	imgheight = im.shape[1]
	starttime = time.time()
	x=0
	for i in range(imgheight//height):
		for j in range(imgwidth//width):
			a,b = (j*width, i*height)
			im[a:a+width,b:b+height,:] = findBestMatch(im[a:a+width,b:b+height,:],1)
			x=x+1
	endtime = time.time()
	print("\t"+str(endtime-starttime)+"   "+str(x))

	return im

def mosaic(image, poolsize):
	a = time.time()
	x=replaceEachBlock(image,mosaic_resol,mosaic_resol)
	b = time.time()
	print(b - a)
	return x
	




def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    resol = [500,500]
    xresstart = (720-resol[0])/2
    xresend = 720 - (720-resol[0])/2
    yresstart = (1280-resol[1])/2
    yresend = 1280 - (1280-resol[1])/2

    while True:
    #for i in range(1,100):
        ret_val, img = cam.read()
        
        img = cv2.flip(img, 1)

        img = img[xresstart:xresend,yresstart:yresend,:]
        #print(img.shape)
        #mosaic(img,1)
        cv2.imshow('my webcam', mosaic(img,1))
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

  