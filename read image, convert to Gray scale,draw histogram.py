from PIL import Image
from pylab import *
import numpy as np
import cv2
# read image to array using matplotlib
im = array(Image.open('images/Sydney-Opera-House.jpg'))

# plot the image
imshow(im)

# add title and show the plot
title('Plotting: "Sydney-Opera-House.jpg"')
show()



# load and show an image with Pillow
# load the image
image = Image.open('images/screenshot.jpg')
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
image.show()

plt.hist(im.ravel(),256,[0,256])
plt.title(' histogram of grey scale img of "Sydney-Opera-House.jpg"')
plt.show()


img = cv2.imread('images/mid resolution img.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title(' histogram of RGB img of a ship ')
plt.show()
