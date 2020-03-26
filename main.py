import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split
from helper import *



def main():

    # LOADING DATA
    X, y = sklearn.datasets.make_moons(n_samples=600, shuffle=True, noise=0.3, random_state=None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    y_train = y_train.reshape(1, y_train.shape[0])
    y_test = y_test.reshape(1, y_test.shape[0])
    X_train, X_test = X_train.T, X_test.T

    # MODEL PARAMETERS
    layers_dims = [X_train.shape[0], 6, 4, 2, 1]  # layers of neural the network
    optimizers = {'GD': 'GD', 'momentum': 'GD', 'adam': 'adam'}

    # PLOTTING THE RESULTS
    for optimizer in optimizers:
        parameters = model(X_train, y_train, layers_dims, optimizer=optimizer)
        plot_cost_function = predict(parameters, X_test)

        # plots decision boundary
        plt.figure()
        plt.title("Model with " + str(optimizer))
        plot_decision_boundary(lambda x: predict_dec(parameters, x.T), X_train, y_train)

    plt.show()


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=6000, print_cost=True):
    """
    Arguments:
    X - input data tyoe np array
    Y - target, np array: blue dot : 1, red dot : 0)
    layers_dims -size of each layer, list
    learning_rate - the learning rate, type scalar.
    mini_batch_size - the size of a mini batch, type scalar
    beta -momentum hyperparameter
    beta1 - xxponential decay hyperparameter for the past gradients estimates
    beta2 - exponential decay hyperparameter for the past squared gradients estimates
    epsilon - hyperparameter (preventing division by zero in Adam updates)
    num_epochs - number of epochs type i nt
    print_cost - type bool, print the cost every 1000 epochs if true

    Returns:
    parameters - updated parameters, type dictionnary
    """

    L = len(layers_dims)  # number of layers in NN
    costs = []
    t = 0
    seed = 1
    m = X.shape[1]

    parameters = initialize_parameters(layers_dims)

    # initializing the optimizer
    if optimizer == "GD":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # optimization
    for i in range(num_epochs):

        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch

            # fforward propagation
            a, caches = L_model_forward(minibatch_X, parameters)

            cost_total += compute_cost(a, minibatch_Y)

            # backward propagation
            grads = L_model_backward(a, minibatch_Y, caches)

            # parameters updating
            if optimizer == "GD":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)
        cost_avg = cost_total / m
        if print_cost and i % 1000 == 0:
            print("cost after %i epochs: %f" % (i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    # plot the cost
    plt.figure()
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title(" Optimizer : " + str(optimizer) + " \nLearning rate =  " + str(learning_rate))

    return parameters


if __name__ == '__main__':
    main()