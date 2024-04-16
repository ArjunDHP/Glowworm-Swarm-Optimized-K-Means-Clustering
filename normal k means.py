# normal k means
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from google.colab import drive
from PIL import Image
from io import BytesIO

# Connect to Google Drive
drive.mount('/content/drive')

# Function to load an image from Google Drive
def load_image_from_drive(file_path):
    image = cv2.imread(file_path)
    return image

# Function to perform K-Means clustering on the image
def kmeans_clustering(image, n_clusters):
    # Convert to floats instead of the default 8-bit integer coding.
    image = np.array(image, dtype=np.float64) / 255.0

    # Get the dimensions of the image
    w, h, d = original_shape = image.shape

    # Reshape the image into a 2D array
    image_array = image.reshape(-1, d)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_array)

    # Predict color indices for the full image
    labels = kmeans.predict(image_array)

    # Function to recreate the image from the codebook & labels
    def recreate_image(codebook, labels, w, h):
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

    # Convert the image to 8-bit unsigned integer (CV_8U) for display
    img_display = (recreate_image(kmeans.cluster_centers_, labels, w, h) * 255).astype(np.uint8)

    return img_display

# Upload an image from Google Drive
file_path = "/content/thisisfinalch2.jpg"  # Update with your image path
uploaded_image = load_image_from_drive(file_path)

# Specify the number of clusters
num_clusters = 32 # Change the number of clusters as needed

# Perform K-Means clustering on the uploaded image
clustered_image = kmeans_clustering(uploaded_image, num_clusters)

# Display the original and clustered images
#
plt.subplot(111)
plt.axis('off')
plt.imshow(cv2.cvtColor(clustered_image, cv2.COLOR_BGR2RGB))

plt.show()