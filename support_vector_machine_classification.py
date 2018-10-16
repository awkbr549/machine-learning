import numpy
from matplotlib import pyplot


def function(x):
    return x**2


def calculate_loss_c(x, y):
    value = y * function(x)
    if (value >= 1):
        value = 0
    else:
        value = 1 - value
    return value


def calculate_objective(lam, w, n, x, y):
    regularizer = lam * (w ** 2)
    tempSum = 0.0
    for i in range(0, n):
        tempSum += calculate_loss_c(x, y)
    return regularizer + tempSum


# derivative of (lam * (w ** 2)) with respect to w --> 2 * lam * |w|
# derivative of (1 - (y * function(x))) with respect to w (f?) --> -y * x


def calculate_regularizer_derivative(lam, w):
    return 2 * lam * w


def calcualte_loss_derivative(x, y):
    value = y * function(x)
    if (value >= 1):
        value = 0
    else:
        value = -y * x
    return value


def svm_sgd_plot(X, Y):
    # initialize svm weight vector
    w = numpy.zeros(len(X[0]))

    learning_rate = 1.0
    epochs = 100000

    # store misclassification so we can plot how they change over time
    errors = []




# [x, y, bias]
X = numpy.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1]
])

Y = numpy.array([-1, -1, 1, 1, 1])

for d, sample in enumerate(X):
    symbol = "+"
    if d < 2:
        symbol = "_"
    pyplot.scatter(sample[0], sample[1], s=120, marker=symbol)

pyplot.plot([-2, 6], [6, 0.5])
pyplot.show()
