import numpy as np
# from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

# def convolve2d(X, W):
#     t0 = datetime.now()
#     # Dimensions of X
#     n1, n2 = X.shape
#     # Dimensions of W (filter)
#     m1, m2 = W.shape
#     row = n1 + m1 -1
#     col = n2 + m2 - 1
#     Y = np.zeros((row, col))
#     for i in range(row):
#         for ii in range(m1):
#             for j in range(col):
#                 for jj in range(m2):
#                     if i>=ii and j>=jj and i-ii < n1 and j-jj<n2:
#                         Y[i, j] += W[ii, jj]*X[i-ii, j-jj]

#     print("Elapsed time: ", (datetime.now() - t0))
#     return Y

# def convolve2d(X, W):
#     t0 = datetime.now()
#     # Dimensions of X
#     n1, n2 = X.shape
#     # Dimensions of W (filter)
#     m1, m2 = W.shape
#     row = n1 + m1 -1
#     col = n2 + m2 - 1
#     Y = np.zeros((row, col))
#     for i in range(n1):
#         for j in range(n2):
#             Y[i:i+m1, j:j+m1] += X[i, j]*W

#     print("Elapsed time: ", (datetime.now() - t0))
#     return Y

# def convolve2d(X, W):
#     t0 = datetime.now()
#     # Dimensions of X
#     n1, n2 = X.shape
#     # Dimensions of W (filter)
#     m1, m2 = W.shape
#     row = n1 + m1 -1
#     col = n2 + m2 - 1
#     Y = np.zeros((row, col))
#     for i in range(n1):
#         for j in range(n2):
#             Y[i:i+m1, j:j+m1] += X[i, j]*W
#     ret = Y[m1//2:-m1//2+1,m2//2:-m2//2+1]
#     print(ret.shape, ", ", X.shape)
#     return ret

# Theano convolution produces a smaller output of n-m+1
# smaller than input
def convolve2d(X, W):
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
    for i in range(n1):
        for j in range(n2):
            Y[i:i+m1,j:j+m2] += X[i,j]*W
    ret = Y[m1-1:-m1+1,m2-1:-m2+1]
    return ret

img = mpimg.imread("raw_data/lena.png")
plt.imshow(img)
plt.show()

# Make the image black and white, as convolve2d is for 2d matrices only
# 512x512
bw = img.mean(axis = 2)
print("img: ", img.shape, "bw: ", bw.shape)
plt.imshow(bw, cmap = "gray")
plt.show()

# Create gaussian filter (20x20)
W = np.zeros((20, 20))



for i in range(20):
    for j in range(20):
        # Squared distance from the center
        dist = (i - 9.5)**2 + (j - 9.5)**2
        # Negative exponential of the dist
        W[i, j] = np.exp(-dist/50)
print("W: ", W)
plt.imshow(W, cmap="gray")
plt.show()

# Convolution
out = convolve2d(bw, W)
plt.imshow(out, cmap="gray")
plt.show()
# 531x531 - Different shape from inputs
print("out: ", out.shape)
# Make them the same
# out = convolve2d(bw, W, mode="same")
# # Output of 512x512, with the black border cropped out
# plt.imshow(out, cmap="gray")
# plt.show()
# print("out: ", out.shape)

# # Gaussian blur for 3 channels
# out3 = np.zeros(img.shape)
# for i in range(3):
#     out3[:,:,i] = convolve2d(img[:,:,i], W, mode='same')
# plt.imshow(out3)
# plt.show()
# print("out3: ", out3.shape)
