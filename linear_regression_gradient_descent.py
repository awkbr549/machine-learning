import matplotlib.pyplot as pyplot
import numpy

def function(x):
    '''m = 100
    b = -10
    return (m * x) + b'''
    a = 7
    b = 3
    c = -11
    return (a * (x**2)) + (b * x) + c


'''def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[0][i]
        y = points[1][i]
        totalError += (y - ((m * x) + b)) ** 2
    return totalError / float(len(points))
    '''

def compute_error_for_line_given_points(a, b, c, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[0][i]
        y = points[1][i]
        total_error += (y - ((a * (x**2)) + (b * x) + c)) ** 2
    return total_error / float(len(points))


'''def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
    return b, m'''


def gradient_descent_runner(points, start_a, start_b, start_c, learning_rate, num_iterations):
    a = start_a
    b = start_b
    c = start_c
    for i in range(num_iterations):
        a, b, c = step_gradient(a, b, c, points, learning_rate)
        if (i % 100 == -1):
            print("\ta: " + str(a))
            print("\tb: " + str(b))
            print("\tc: " + str(c))
            print()
    return (a, b, c)


'''def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[0][i]
        y = points[1][i]

        b_gradient += -(2/N) * (y - (m_current * x) - b_current)
        m_gradient += -(2/N) * (y - (m_current * x) - b_current) * x

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return new_b, new_m'''


def step_gradient(a_current, b_current, c_current, points, learning_rate):
    a_grad = 0
    b_grad = 0
    c_grad = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[0][i]
        y = points[1][i]

        a_grad += (-2/N) * (y - (a_current * (x ** 2)) + (b_current * x) + c_current) * (x ** 2)
        # b_grad += (-2/N) * (y - (a_current * (x ** 2)) + (b_current * x) + c_current) * x
        # c_grad += (-2/N) * (y - (a_current * (x ** 2)) + (b_current * x) + c_current)

    new_a = a_current - (learning_rate * a_grad)
    new_b = b_current - (learning_rate * b_grad)
    new_c = c_current - (learning_rate * c_grad)

    return (new_a, new_b, new_c)


'''def main():
    N = 1000
    random_x = []
    random_y = []
    for i in range(0, N):
        random_x.append(i/10.0)
        random_y.append(function(i/10.0))

    learning_rate = 0.0001
    num_iterations = 10**3
    print("descending...")
    b, m = gradient_descent_runner([random_x, random_y], 0, 0, learning_rate, num_iterations)
    print("b: " + str(b))
    print("m: " + str(m))
    print("error: " + str(compute_error_for_line_given_points(b, m, [random_x, random_y])))'''


def main():
    N = 1000
    random_x = []
    random_y = []
    for i in range(int(-N/2), int(N/2)):
        random_x.append(i/10.0)
        random_y.append(function(i/10.0))

    # pyplot.plot(random_x, random_y, 'ro')
    # pyplot.show()

    learning_rate = 0.0000001
    num_iterations = 10**7
    print("descending...")
    (a, b, c) = gradient_descent_runner([random_x, random_y], 0, 0, 0, learning_rate, num_iterations)
    print("a: " + str(a))
    print("b: " + str(b))
    print("c: " + str(c))
    print("error: " + str(compute_error_for_line_given_points(a, b, c, [random_x, random_y])))


main()
