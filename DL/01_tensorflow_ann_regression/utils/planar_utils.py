import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

# def points_within_circle(radius, 
#                          center=(0, 0),
#                          number_of_points=100):
#     center_x, center_y = center
#     r = radius * np.sqrt(np.random.random((number_of_points,)))
#     theta = np.random.random((number_of_points,)) * 2 * np.pi
#     x = center_x + r * np.cos(theta)
#     y = center_y + r * np.sin(theta)
#     return x, y

# def load_separable_dataset():
    
#     x_11, x_12 = points_within_circle(1.6, (5, 2), 100)
#     x_1 = np.stack((x_11, x_12), axis=1)
#     y_1 = np.zeros(x_1.shape[0])
    
#     x_21, x_22 = points_within_circle(1.9, (2, 5), 100)
#     x_2 = np.stack((x_21, x_22), axis=1)
#     y_2 = np.ones(x_2.shape[0])
    
#     X = np.vstack((x_1, x_2))
#     Y = np.hstack((y_1, y_2))

#     return X, Y

def load_separable_dataset(radius=[1.9,1.6], 
                           center=[(5, 2), (2,5)],
                           number_of_points=100):
    Xs = []
    Ys = []
    for i in range(len(radius)):
        center_x, center_y = center[i]
        r = radius[i] * np.sqrt(np.random.random((number_of_points,)))
        theta = np.random.random((number_of_points,)) * 2 * np.pi
        x_1 = center_x + r * np.cos(theta)
        x_2 = center_y + r * np.sin(theta)
        x = np.stack((x_1, x_2), axis=1)
        y = np.ones(x.shape[0])*i
        
        Xs.append(x)
        Ys.append(y)
    
    X = np.array([])
    Y = np.array([])
    for j in range(len(radius)):
        X = np.vstack((X, Xs[j])) if X.size else Xs[j]
        Y = np.hstack((Y, Ys[j])) if Y.size else Ys[j]

    return X, Y

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    # no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles#, no_structure


def load_non_separable_dataset():
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, Y

# lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()