from PIL import Image
import cv2
import tensorflow as tf
import pickle
import datetime
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas import DataFrame 
from sklearn import datasets 
from sklearn.mixture import GaussianMixture 
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import sys
from matplotlib import animation


infile = open(sys.argv[1]+'indexes','rb')
indexes = pickle.load(infile)
infile.close()

index = np.array([a[0] for a in indexes])
pool1 = np.array([a[1] for a in indexes])
pool3 = np.array([a[2] for a in indexes])
pool5 = np.array([a[3] for a in indexes])

pool1flat = pool1.reshape(pool1.shape[0], pool1.shape[1]*pool1.shape[2]*pool1.shape[3])
pool3flat = pool3.reshape(pool3.shape[0], pool3.shape[1]*pool3.shape[2]*pool3.shape[3])
pool5flat = pool5.reshape(pool5.shape[0], pool5.shape[1]*pool5.shape[2]*pool5.shape[3])

fig = pyplot.figure()
ax = Axes3D(fig)

# ids = np.random.choice(pool1flat.shape[0], 1000, replace=False)  
# pool1flat = pool1flat[ids]

x = [a[0] for a in pool1flat]
y = [a[1] for a in pool1flat]
z = [a[2] for a in pool1flat]

# #ax.scatter(x, y, z,s=1)
# ax.set_xlabel('Red', fontsize=18)
# ax.set_ylabel('Blue', fontsize=18)
# ax.set_zlabel('Green', fontsize=18)
# ax.set_title('Main')
# #pyplot.show()
# def init():
# 	ax.scatter(x, y, z,c=np.column_stack((x,y,z))/256.0,s=1)
# 	return fig,

# def animate(s):
#     ax.view_init(30, s)
#     return fig,

# anim = animation.FuncAnimation(fig, animate, init_func=init,frames=360, interval=400, blit=True)
# print('saving animation')
# anim.save(sys.argv[1]+'basic_animation.gif', writer='imagemagick', fps=60)

# pyplot.show()

components = 2
d = pd.DataFrame(pool1flat) 
gmm = GaussianMixture(n_components = components) 
print("fitting")
gmm.fit(d) 

# Assign a label to each sample 
print("predicting")
labels = gmm.predict(d) 
d['labels']= labels 
colors = ['r','black','g','blue','orange','black','cyan','brown']
for i in range(components):
	d0 = d[d['labels']== i]
	fig = pyplot.figure()
	ax = Axes3D(fig)
	ax.set_title("Cluster "+str(i+1))
	ax.set_xlabel('Red', fontsize=18)
	ax.set_ylabel('Blue', fontsize=18)
	ax.set_zlabel('Green', fontsize=18)
	def init():
		ax.scatter(d0[0], d0[1],d0[2], c = np.column_stack((d0[0], d0[1],d0[2]))/256.0,s=1)
		return fig,
	def animate(s):
	    ax.view_init(30, s)
	    return fig,

	allImagesinThisGrp = index[d['labels']== i]
	for image in allImagesinThisGrp:
		print("cp "+sys.argv[1]+"/"+str(image)+".jpg clusters/"+str(i)+"/"+str(image)+".jpg ")

	anim = animation.FuncAnimation(fig, animate, init_func=init,frames=360, interval=400, blit=True)
	print('saving animation')
	
	anim.save(sys.argv[1]+'_cluster_animation_'+str(i)+'.gif', writer='imagemagick', fps=30)



