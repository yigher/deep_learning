import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("raw_data/lena.png")
# Make the image black and white, as convolve2d is for 2d matrices only
# 512x512
bw = img.mean(axis = 2)
print("img: ", img.shape, "bw: ", bw.shape)

# Sobel operator - defined for the 2 directions (x and y), 
# and approximate the gradient at each point in the image
Hx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], dtype=np.float32)
print("Hx: ", Hx)
# Tranpose of Hx [[-1,-2,-1], [0,0,0], [1,2,1]]
Hy = np.transpose(Hx)
print("Hy: ", Hy)
# Do a convolution to get Gradients Gx and Gy
# Detect horizontal edges
Gx = convolve2d(bw, Hx)
plt.imshow(Gx, cmap='gray')
plt.show()
# Detect vertical edges
Gy = convolve2d(bw, Hy)
plt.imshow(Gy, cmap='gray')
plt.show()
# Gx and Gy are vectors. Magnitude and direction can be calculated
# G = edge detected for x and y axis
G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
plt.show()
# Plot the gradient direction in the image
theta = np.arctan2(Gy, Gx)
plt.imshow(theta, cmap='gray')
plt.show()
