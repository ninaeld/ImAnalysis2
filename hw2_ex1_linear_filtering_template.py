# Image Analysis Homework 2 ex 1
# Nina Eldridge

""" 1 Linear filtering """

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import time
import pdb

img = plt.imread('cat.jpg').astype(np.float32)

plt.imshow(img)
plt.axis('off')
plt.title('original image')
plt.show()

# 1.1
def boxfilter(n):
    # this function returns a box filter of size nxn
    box_filter = np.array((n**2) * [1/(n**2)]).reshape((n,n))
    return box_filter

# 1.2
# Implement full convolution
def myconv2(image, filt):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two images. DO
    # NOT USE THE BUILT IN SCIPY CONVOLVE within this function. You should code your own version of the
    # convolution, valid for both 2D and 1D filters.
    # INPUTS
    # @ image         : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)
    image_height, image_width = image.shape
    #print("Shape of image is " +str(image.shape))
    filter_height, filter_width = filt.shape
    conv_image = image

    # do the padding for zeros
    added_columns = np.zeros(image_height)
    # adds columns with zeros to either side of the image
    for i in range(int(filter_width/2)):
        conv_image = np.insert(conv_image, 0, added_columns, axis=1)
        conv_image = np.insert(conv_image,conv_image.shape[1], added_columns, axis=1)
    # add rows with zeros on top and bottom of image
    added_rows = np.zeros(image_width+filter_width-1)
    for j in range(int(filter_height/2)):
        conv_image = np.vstack((added_rows, conv_image))
        conv_image = np.vstack((conv_image, added_rows))

    conv_height, conv_width = conv_image.shape
    working_conv = conv_image.copy()
    #print("Shape of working conv is " +str(working_conv.shape))
    # create convolved image filled with zeros
    #conv_image = np.zeros((image_width+filter_width-1, image_height+filter_height-1))
    # s goes over the rows
    for s in range((conv_height-filter_height)):
        # t goes over the columns
        for t in range(0, (conv_width-filter_width)):
            # slices the part of the image that the filter is over
            part_image = working_conv[s:(s+filter_height), t:(t+filter_width)]
            #print(part_image.shape)
            # applies the filter to the part of the image
            value = np.sum(np.multiply(part_image, filt))
            #print(value)
            # updates the value at the right location in the picture
            conv_image[s+(int(filter_height/2)), t+(int(filter_width/2))] = value

    return conv_image


# 1.3
# create a boxfilter of size 11 and convolve this filter with your image - show the result
bsize = 11
# creates the boxfilter with the function
my_boxfilter = boxfilter(bsize)
# convolves the image with the created boxfilter
convolved_image = myconv2(img, my_boxfilter)
plt.imshow(convolved_image)
plt.title('box filtered image')
plt.show()

# 1.4
# create a function returning a 1D gaussian kernel
def gauss1d(sigma, filter_length=20):
    pass
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    ### your code should go here ###

    #return gauss_filter


# 1.5
# create a function returning a 2D gaussian kernel
def gauss2d(sigma, filter_size=20):
    pass
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_size   : integer denoting the filter size, default is 10
    # OUTPUTS
    # @ gauss2d_filter  : 2D gaussian filter

    ### your code should go here ###

    #return gauss2d_filter

# Display a plot using sigma = 3
sigma = 3

### your code should go here ###


# 1.6
# Convoltion with gaussian filter
def gconv(image, sigma):
    pass
    # INPUTS
    # image           : 2d image
    # @ sigma         : sigma of gaussian distribution
    # OUTPUTS
    # @ img_filtered  : filtered image with gaussian filter

    ### your code should go here ###

    #return img_filtered


# run your gconv on the image for sigma=3 and display the result
sigma = 3

### your code should go here ###


# 1.7
# Convolution with a 2D Gaussian filter is not the most efficient way
# to perform Gaussian convolution with an image. In a few sentences, explain how
# this could be implemented more efficiently and why this would be faster.
#
# HINT: How can we use 1D Gaussians?

### your explanation should go here ###

# 1.8
# Computation time vs filter size experiment
size_range = np.arange(3, 100, 5)
t1d = []
t2d = []
for size in size_range:
    pass
    ### your code should go here ###


# plot the comparison of the time needed for each of the two convolution cases
"""plt.plot(size_range, t1d, label='1D filtering')
plt.plot(size_range, t2d, label='2D filtering')
plt.xlabel('Filter size')
plt.ylabel('Computation time')
plt.legend(loc=0)
plt.show()"""
