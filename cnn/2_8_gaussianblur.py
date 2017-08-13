import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
out = convolve2d(bw, W, mode="same")
# Output of 512x512, with the black border cropped out
plt.imshow(out, cmap="gray")
plt.show()
print("out: ", out.shape)

# Gaussian blur for 3 channels
out3 = np.zeros(img.shape)
for i in range(3):
    out3[:,:,i] = convolve2d(img[:,:,i], W, mode='same')
plt.imshow(out3)
plt.show()
print("out3: ", out3.shape)
