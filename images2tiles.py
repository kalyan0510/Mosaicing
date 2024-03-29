from __future__ import print_function
from PIL import Image, ExifTags
import os, sys
import scipy.misc
from math import sin,cos,pi
import numpy as np

# usage: python images2tiles.py inputfolder outputfolder 100

def roation_mat(x,y,angle):
  p = x*int(cos(angle))-y*int(sin(angle))
  q = y*int(cos(angle))+x*int(sin(angle))
  return [p,q]


def rot(angle,img):
  h,w,c = img.shape
  newimg = np.zeros([h,w,c], dtype=np.uint8)
  for i in range(h):
    for j in range(w):
      inew,jnew=roation_mat(i,j,angle)
      newimg[inew,jnew] = img[i,j]
  return newimg

def flip(xoy,img):
  h,w,c = img.shape
  newimg = np.zeros([h,w,c], dtype=np.uint8)
  for i in range(h):
    for j in range(w):
      newimg[i,j] = img[(-1 if xoy=='y' else 1)*i,(-1 if xoy=='x' else 1)*j]
  return newimg




def square_thumb(img, thumb_size):
  THUMB_SIZE = (thumb_size,thumb_size)

  width, height = img.size

  # square it

  if width > height:
    delta = width - height
    left = int(delta/2)
    upper = 0
    right = height + left
    lower = height
  else:
    delta = height - width
    left = 0
    upper = int(delta/2)
    right = width
    lower = width + upper

  img = img.crop((left, upper, right, lower))
  img.thumbnail(THUMB_SIZE, Image.ANTIALIAS)

  return np.array(img)

i = 0

for subdir, dirs, files in os.walk(sys.argv[1]):
  for file in files:
    try:
      file.strip()
      if(file.endswith(".jpg")):
        image = Image.open(os.path.join(subdir, file))
        opImg = square_thumb(image, int(sys.argv[3]))
        newImges = [opImg,flip('x',opImg), rot(pi/2,opImg),rot(3*pi/2,opImg)]
        for img in newImges:
          i = i+1
          #data[str(i)]=imgFile
          scipy.misc.imsave(sys.argv[2]+"/"+str(i)+".jpg", img)
          if i%100 == 0:
            print(i)
    except Exception as e:
      print(e)
      print("Failed for "+str(i))
print("Tota of "+str(i)+"images")
