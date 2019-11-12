from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage

from PIL import Image
from pylab import *

# read image to array
im = array(Image.open('images/water_coins.jpg'))
print(' displaying the dimension of input img : ' , im.shape)

# plot the image
imshow(im)
plt.title('Plotting input image')
#plt.show()


print('before show grayscale')

gray = rgb2gray(im)
plt.imshow(gray, cmap='gray')
plt.title('gray scaled image')
#plt.show()

print('after show grayscale')

print('now performing segmentation using clustering')
'''It’s a 3-dimensional image of shape (192, 263, 3). 3 for RGB channel
For clustering the image using k-means,
we first need to convert it into a 2-dimensional array
whose shape will be (length*width, channels).
here it will be (192*263, 3). 
'''
pic_n = im.reshape(im.shape[0]*im.shape[1], im.shape[2])
print(' displaying the dimension of reshaped img : ' , pic_n.shape)

# the image has been converted to a 2-dimensional array.

n_clusters=4
kmeans = KMeans(n_clusters, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(im.shape[0], im.shape[1], im.shape[2])
plt.imshow(cluster_pic.astype('uint8'))
plt.title('segmented image using %d means clustering ' % n_clusters)
plt.show()
