import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
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
import random, sys
import random


infile = open(sys.argv[1],'rb')
indexes = pickle.load(infile)
infile.close()

print(len(indexes))
indexes = random.sample(indexes,k=len(indexes)/100)
print(len(indexes))

index = np.array([a[0] for a in indexes])
pool1 = np.array([a[1] for a in indexes])
# pool3 = np.array([a[2] for a in indexes])
# pool5 = np.array([a[3] for a in indexes])

pool1flat = pool1.reshape(pool1.shape[0], pool1.shape[1]*pool1.shape[2]*pool1.shape[3])
# pool3flat = pool3.reshape(pool3.shape[0], pool3.shape[1]*pool3.shape[2]*pool3.shape[3])
# pool5flat = pool5.reshape(pool5.shape[0], pool5.shape[1]*pool5.shape[2]*pool5.shape[3])


spectral=cluster.SpectralClustering(n_clusters=3,eigen_solver='arpack',affinity="nearest_neighbors")

spectral.fit(pool1flat)
if hasattr(spectral,'labels_'):
	y_pred=spectral.labels_.astype(np.int)
else:
	y_pred=spectral.predict(X)

colors=np.array(list(islice(cycle(['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00']),int(max(y_pred)+1))))

fig = pyplot.figure()
ax = Axes3D(fig)
ax.set_xlabel('Red', fontsize=18)
ax.set_ylabel('Blue', fontsize=18)
ax.set_zlabel('Green', fontsize=18)

ax.scatter(pool1flat[:, 0], pool1flat[:, 1],pool1flat[:,2], c =colors[y_pred],s=1) 
plt.show() 