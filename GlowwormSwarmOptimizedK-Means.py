import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def load_image(file_path):
    # Load an image from a file and return it as a NumPy array
    image = plt.imread(file_path)
    return image

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x


def glowworm_optimization(input_image, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5

    N = opts['N'] #number of worms
    max_iter = opts['T']
    num_clusters = opts['num_clusters']  # Number of clusters for k-means


    # Dimension
    dim = np.prod(input_image.shape)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xdb = np.zeros([1, dim], dtype='float')
    fitD = float('inf')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i, 0] = fitness(input_image, Xbin[i, :], opts)
            if fit[i, 0] < fitD:
                Xdb[0, :] = X[i, :]
                fitD = fit[i, 0]

        # Store result
        curve[0, t] = fitD.copy()
        print("Iteration:", t + 1)
        print("Best (Glowworm Optimization):", curve[0, t])
        t += 1



    # Best feature subset
    Gbin = binary_conversion(Xdb, thres, 1, dim)
    Gbin = Gbin.reshape(input_image.shape)

    # Use the best centroids for k-means clustering
    flattened_image = input_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(flattened_image)
    Gbin = kmeans.cluster_centers_.reshape(1, -1, 3)
    labels = kmeans.labels_

    # Reconstruct the segmented image
    clustered_image = Gbin[0, labels].reshape(input_image.shape)

    # Normalize segmented_image to the range [0, 1]
    clustered_image = (clustered_image - clustered_image.min()) / (clustered_image.max() - clustered_image.min())

    # Create dictionary
    glowworm_data = {'c': curve, 'clustered_image': clustered_image}

    return glowworm_data

# Map cluster labels to colors and update the clustered image
def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) *np.random.rand()

    return X

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin

def fitness(input_image, solution, opts):

    # Reshape solution and input_image to match shape
    solution = solution.reshape(input_image.shape)
    # Calculate fitness based on the difference between the solution and input_image
    diff = (solution - input_image) ** 2
    return np.sum(diff)

if __name__ == "__main__":
    # Load the input image
    input_image = load_image("/content/ch1-crater.jpg")

    # Optimization parameters
    opts = {'N': 20, 'T': 1, 'num_clusters': 32}

    # Apply Glowworm optimization and K-means clustering
    result = glowworm_optimization(input_image, opts)
    clustered_image = result['clustered_image']

    # Save the segmented image
    plt.imsave("clustered_image.jpeg", clustered_image)
