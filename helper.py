import numpy as np
import matplotlib.pyplot as plt
import math


def sigmoid(Z):

   A = 1/(1+np.exp(-Z))
   cache = Z
   return A, cache


def relu(Z):

   A = np.maximum(0,Z)
   cache = Z

   return A, cache


def relu_backward(dA, cache):

   Z = cache
   dZ = np.array(dA, copy=True)
   dZ[Z <= 0] = 0

   return dZ


def sigmoid_backward(dA, cache):

   Z = cache
   s = 1/(1+np.exp(-Z))
   dZ = dA * s * (1-s)

   return dZ


def plot_decision_boundary(model, X, y):
    """
    Arguments:
    model - model with computed parameters
    X- input, np array
    y - target, np array

    Returns:
    None, plot the results
    """

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
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.ylabel('x2')
    plt.xlabel('x1')
    color = y
    data = np.concatenate((X, y), axis=0)
    plt.scatter(data[0, :], data[1, :], c=data[2, :], cmap=plt.cm.Spectral)


def predict(parameters, X):
    """
    Arguments:
    parameters - parameters, type dict

    Returns:
    X - intput, type np array
    """
    m = X.shape[1]  # number of layers in the neural network
    p = np.zeros((1, m))

    # forward propa
    probas, caches = forward_propagation(X, parameters)
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


def initialize_parameters(layers_dims):
    """
    Arguments:
    layer_dims - size of each layer, type list

    Returns:
    parameters - parameters, type dict

    """

    np.random.seed()
    parameters = {}
    L = len(layers_dims) - 1  # number of layers

    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
            2. / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def compute_cost(AL, Y):
    """

    Arguments:
    AL - probability vector corresponding to label predictions, type np array
    Y - target vector, numpy array

    Returns:
    cost - cross-entropy cost
    """

    m = Y.shape[1]
    cost = -np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL))) / m

    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    """
    Arguments:
    dZ -gradient of the costwith respect to the linear output, type np array
    cache - values from the forward propagation in the current layer, type tupple

    return:
    dA_prev - gradient of the cost with respect to the activation
    dW -gradient of the cost with respect to W (current layer l), type np array
    db - gradient of the cost with respect to b (current layer l),type np array
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = dZ @ A_prev.T / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T @ dZ

    return dA_prev, dW, db


def forward_propagation(X, parameters):
    """

    Arguments:
    X - data, type np array
    parameters -output of initialize_parameters_deep()

    Returns:
    AL - last post-activation value
    caches - contains  every cache of linear_activation_forward(), type dict
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
                                             activation='relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches


def linear_activation_forward(A_prev, W, b, activation):
    """

    Arguments:
    A_prev - activations from previous layer, np array
    W - weghts matrix: type np array
    b - bias vector, type np array
    activation - activation to be used in the layer, type string

    return:
    A -post-activation value, np array
    cache - contains linear_cache" and "activation_cache", type tupple
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def linear_activation_backward(dA, cache, activation):
    """
    Arguments:
    dA - postactivation gradient for current layer l
    cache -contains linear_cache and activation_cache, type tupple
    activation - activation to be used in the layer, type string

    return:
    dA_prev -gradient of the costwith respect to the activation (of the previous layer l-1), type np array
    dW - gradient of the cost xith respect to W (current layer l), type np array
    db - gradient of the cost with respect to b(current layer l), type np array
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def linear_forward(A, W, b):
    """
    Arguments:
    A -activations from previous layer, type np array
    W - weights matrix: type numpy array
    b -bias vector, type np array

    Returns:
    Z - pre-activation parameter, np array
    cache- tuple (A, W, b) for backward pass computing
    """

    Z = W @ A + b

    cache = (A, W, b)

    return Z, cache


def initialize_velocity(parameters):
    """
    Arguments:
    parameters - parameters, type dict

    Returns:
    v - current velocity, type dict

    """

    L = len(parameters) // 2
    v = {}

    # initialising velocity
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape))
        v["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape))

    return v


def initialize_adam(parameters):
    """
    Arguments:
    parameters - parameters, type dict

    Returns:
    v- exponentially weighted average of the gradient, type dict
    s- exponentially weighted average of squared gradient, type dict

    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v, s


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Arguments:
    X -input data, type p array
    Y -target vector, type np array
    mini_batch_size - mini-batches size, type int

    Returns:
    mini_batches - list of synchronous (mini_batch_X, mini_batch_Y), type list
    """

    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size in partition
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * (k):  mini_batch_size * (k + 1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * (k): mini_batch_size * (k + 1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:  # last mini-batch < mini_batch_size
        s = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:, mini_batch_size * k: mini_batch_size * (k) + s]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k: mini_batch_size * (k) + s]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Arguments:
    parameters - paraeters, type dict
    grads - gradients, python dict
    learning_rate - the learning rate, type float

    Returns:
    parameters - python dict containing the updated parameters
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Arguments:
    parameters -parmeters, type dict

    grads - gradients, type dict

    v - current velocity, type dict

    beta -the momentum hyperparameter, type float
    learning_rate -the learning rate, type float

    Returns:
    parameters - updated parameters, type dict
    v - updated velocities, type dict
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    for l in range(L):
        # computes velocities
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        # updates parameters
        parameters["W" + str(l + 1)] -= learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * v["db" + str(l + 1)]

    return parameters, v


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                            beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Arguments:
    parameters -paramters, python dict such as:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -gradrients, type dict such as;
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v - adam variable, moving average of the first gradient, type dict
    s - Adam variable, moving average of the squared gradient, type dict
    learning_rate - the learning rate, type float
    beta1 - exponential decay hyperparameter for the first moment estimate
    beta2 - exponential decay hyperparameter for the second moment estimate
    epsilon - hyperparameter to avoid /0

    Returns:
    parameters - python dictionary containing your updated parameters
    v - Adam variable, moving average of the first gradient, type dict
    s - Adam variable, moving average of the squared gradient, type dict
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # updates parameters
    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1)

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * grads['dW' + str(l + 1)] * grads[
            'dW' + str(l + 1)]
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * grads['db' + str(l + 1)] * grads[
            'db' + str(l + 1)]

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2)

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / (
                    (s_corrected["dW" + str(l + 1)]) ** 0.5 + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / (
                    (s_corrected["db" + str(l + 1)]) ** 0.5 + epsilon)

    return parameters, v, s


def L_model_forward(X, parameters):
    """
    Arguments:
    X -data, numpy array of shape (input size, number of examples)
    parameters - parameters, type dict

    Returns:
    AL -last post-activation value
    caches - list of caches containing every cache of linear_activation_forward()
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
                                             activation='relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches



def L_model_backward(AL, Y, caches):
    """
    Arguments:
    AL -robability vector (output of the forward propagation), type np array
    Y - target vector, type numpy array
    caches - list of caches containing: \
        every cache of linear_activation_forward() with "relu"
        the cache of linear_activation_forward() with "sigmoid"

    Returns:
    grads- gradients, type dictionnary
    """
    grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation='sigmoid')
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation='relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



def predict_dec(parameters, X):
    """
    Arguments:
    parameters - parameters, type dictionnary
    X -input data, type numpy array

    Returns
    predictions- vector of predictions of the model (red: 0 / blue: 1), type np array
    """

    # Predict using forward propagation and a classification threshold of 0.5
    a, cache = forward_propagation(X, parameters)
    predictions = (a > 0.5)
    return predictions
