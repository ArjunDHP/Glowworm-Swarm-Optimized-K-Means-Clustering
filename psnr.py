import math
import cv2
import numpy as np

original = cv2.imread("/content/chan2jpg.jpg")
contrast = cv2.imread("/content/ch364.png", 1)

# Resize the 'contrast' image to match the dimensions of 'original' image
contrast = cv2.resize(contrast, (original.shape[1], original.shape[0]))

def psnr(img1, img2):
    mse = np.mean(np.square(np.subtract(img1.astype(np.int16),
                                        img2.astype(np.int16))))
    if mse == 0:
        return np.Inf
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

psnr_value = psnr(original, contrast)
mse_value = mse(original, contrast)

print(f"PSNR: {psnr_value}")
print(f"MSE: {mse_value}")
import numpy as np
from sklearn.metrics import adjusted_rand_score
import cv2

# Load the original and clustered images
original = cv2.imread("/content/chan2jpg.jpg", 0)
clustered = cv2.imread("/content/ch364.png", 0)

# Resize the 'clustered' image to match the dimensions of 'original' image
clustered = cv2.resize(clustered, (original.shape[1], original.shape[0]))

# Flatten the images to 1D arrays
original_flatten = original.flatten()
clustered_flatten = clustered.flatten()

# Calculate the Rand Index
rand_index = adjusted_rand_score(original_flatten, clustered_flatten)

print(f"Rand Index: {rand_index}")
import cv2
from skimage.metrics import structural_similarity as ssim

# Load the original and clustered images
original = cv2.imread("/content/chan2jpg.jpg", cv2.IMREAD_GRAYSCALE)
clustered = cv2.imread("/content/ch364.png", cv2.IMREAD_GRAYSCALE)

# Resize the 'clustered' image to match the dimensions of 'original' image
clustered = cv2.resize(clustered, (original.shape[1], original.shape[0]))

# Calculate SSIM
ssim_value = ssim(original, clustered)

print(f"SSIM: {ssim_value}")