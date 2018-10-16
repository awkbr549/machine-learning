import numpy
import networkx
from matplotlib import pyplot


def available_actions(state):
    current_state_row = R[state, ]
    av_act = numpy.where(current_state_row >= 0)[1]
    return av_act


def available_actions_with_enviro_help(state):
    current_state_row = R[state, ]
    av_act = numpy.where(current_state_row >= 0)[1]
    # if there are multiple routes, dis-favor anything negative
    env_pos_row = enviro_matrix[state, av_act]
    if (numpy.sum(env_pos_row < 0)):
        # can we remove the negative directions from av_act?
        temp_av_act = av_act[numpy.array(env_pos_row)[0] >= 0]
        if (len(temp_av_act) > 0):
            av_act = temp_av_act
    return av_act


def sample_next_action(available_act):
    next_action = int(numpy.random.choice(available_act, 1))
    return next_action


def collect_environmental_data(action):
    found = []
    if (action in bees):
        found.append('b')
    if (action in smoke):
        found.append('s')
    return found


def update(current_state, action, gamma):
    max_index = numpy.where(Q[action, ] == numpy.max(Q[action, ]))[1]

    if (max_index.shape[0] > 1):
        max_index = int(numpy.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    Q[current_state, action] = R[current_state, action] + (gamma * max_value)
    #print('max_value', Q[current_state, action])

    # environmental data
    environment = collect_environmental_data(action)
    if ('b' in environment):
        enviro_matrix[current_state, action] += 1
    if ('s' in environment):
        enviro_matrix[current_state, action] -= 1

    result = None
    if (numpy.max(Q) > 0):
        result = numpy.sum(100 * Q / numpy.max(Q))
    else:
        result = 0
    return result


# map cell to cell, add circular cell to goal point
points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]
'''
points_list = []
for i in range(0, 30):
    points_list.append((numpy.random.randint(0, 10), numpy.random.randint(0, 10)))
'''
goal = 7

# plotting the cell-to-cell network
'''
G = networkx.Graph()
G.add_edges_from(points_list)
pos = networkx.spring_layout(G)
networkx.draw_networkx_nodes(G, pos)
networkx.draw_networkx_edges(G, pos)
networkx.draw_networkx_labels(G, pos)
'''
bees = [2]
smoke = [4, 5, 6]
G = networkx.Graph()
G.add_edges_from(points_list)
mapping = {
    0:'Start',
    1:'1',
    2:'2 - Bees',
    3:'3',
    4:'4 - Smoke',
    5:'5 - Smoke',
    6:'6 - Smoke',
    7:'7 - Beehive'
}
H = networkx.relabel_nodes(G, mapping)
pos = networkx.spring_layout(H)
networkx.draw_networkx_nodes(H, pos, node_size=[200,200,200,200,200,200,200,200])
networkx.draw_networkx_edges(H, pos)
networkx.draw_networkx_labels(H, pos)
pyplot.show()

# creating a matrix representation of the cell-to-cell paths
MATRIX_SIZE = 8
R = numpy.matrix(numpy.ones(shape = (MATRIX_SIZE, MATRIX_SIZE)))
R *= -1
print("R:")
print(R)

# assign 0's to viable paths and 100's to goal paths
for point in points_list:
    if (point[0] == goal or point[1] == goal):
        R[point[0], point[1]] = 100
        R[point[1], point[0]] = 100
    else:
        R[point[0], point[1]] = 0
        R[point[1], point[0]] = 0
R[goal, goal] = 100 # add goal point round trip
print()
print("R")
print(R)

# initialization
Q = numpy.matrix(numpy.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# environmental data
enviro_matrix = numpy.matrix(numpy.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# training
gamma = 0.8
threshold = 0.1
counter = 0
new_score = prev_score = 0.0
scores = []
for i in range(0, 1000):
    current_state = numpy.random.randint(0, int(Q.shape[0]))
    available_act = available_actions_with_enviro_help(current_state)
    action = sample_next_action(available_act)
    new_score = update(current_state, action, gamma)

    #scores.append(new_score)
    delta_score = new_score - prev_score
    scores.append(delta_score)

    if (delta_score < threshold):
        counter += 1
    else:
        counter = 0
    if (counter >= 100):
        print("Iterations: " + str(i))
        break
    prev_score = new_score

print("Trained Q matrix:")
print(100 * Q / numpy.max(Q))

# print environmental matrices
print('Environmental Matrix')
print(enviro_matrix)

# testing
current_state = 0
steps = [current_state]
while current_state != 7:

    next_step_index = numpy.where(Q[current_state, ] == numpy.max(Q[current_state, ]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(numpy.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)

    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)

pyplot.plot(scores)
pyplot.show()