import numpy
from matplotlib import pyplot

# [x, y, bias]
X = numpy.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1]
])

Y = numpy.array([-1, -1, 1, 1, 1])

for d, sample in enumerate(x):
    symbol = "+"
    if d < 2:
        symbol = "_"
    pyplot.scatter(sample[0], sample[1], s=120, marker=symbol, linewidth=2)

pyplot.plot([-2, 6], [6, 0.5])
pyplot.show()