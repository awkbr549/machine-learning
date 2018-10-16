import gym
import random
import numpy
from statistics import mean, median, mode
from collections import Counter
import time

def update(current_state, action, gamma, Q, R):
    result = 0
    try:
        #print("1")
        max_value = max(Q[int((current_state[0] + X_MAX) / X_ACCURACY)][
            int((current_state[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
            int((current_state[2] + THETA_MAX) / THETA_ACCURACY)][
            int((current_state[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][0],
                        Q[int((current_state[0] + X_MAX) / X_ACCURACY)][
            int((current_state[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
            int((current_state[2] + THETA_MAX) / THETA_ACCURACY)][
            int((current_state[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][1])

        #print("2")
        result = R[int((current_state[0] + X_MAX) / X_ACCURACY)][
            int((current_state[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
            int((current_state[2] + THETA_MAX) / THETA_ACCURACY)][
            int((current_state[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][action] + (gamma * max_value)
        Q[int((current_state[0] + X_MAX) / X_ACCURACY)][
            int((current_state[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
            int((current_state[2] + THETA_MAX) / THETA_ACCURACY)][
            int((current_state[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][action] = result
    except (IndexError):
        show_problems(current_state)

    #print('max_value', Q[current_state, action])

    #print("3")
    return result, Q, R


def show_problems(observation):
    print(observation)
    print(X_MATRIX_SIZE)
    print(int((observation[0] + X_MAX) / X_ACCURACY))
    print(X_DOT_MATRIX_SIZE)
    print(int((observation[1] + X_DOT_MAX) / X_DOT_ACCURACY))
    print(THETA_MATRIX_SIZE)
    print(int((observation[2] + THETA_MAX) / THETA_ACCURACY))
    print(THETA_DOT_MATRIX_SIZE)
    print(int((observation[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY))
    exit()


def initial_population(initial_games, Q, R):
    for _ in range(0, initial_games):
        env.reset()
        prev_observation = None
        for _ in range(0, goal_steps):
            #env.render()
            action = random.randrange(0, 2)
            #observation --> self.state = (x, x_dot, theta, theta_dot)
            observation, reward, done, info = env.step(action)
            if (not prev_observation is None):
                try:
                    Q[int((prev_observation[0] + X_MAX) / X_ACCURACY)][
                        int((prev_observation[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
                        int((prev_observation[2] + THETA_MAX) / THETA_ACCURACY)][
                        int((prev_observation[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][
                        action] += reward

                    R[int((prev_observation[0] + X_MAX) / X_ACCURACY)][
                        int((prev_observation[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
                        int((prev_observation[2] + THETA_MAX) / THETA_ACCURACY)][
                        int((prev_observation[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][
                        action] = reward
                except (IndexError):
                    show_problems(observation)

            prev_observation = observation
            #time.sleep(0.1)

            if (done):
                break
    return Q, R

    #print("Average accepted score: " + str(mean(accepted_scores)))
    #print("Median accepted score: " + str(median(accepted_scores)))
    #return Q, R


def available_actions(state, Q):
    current_state_row = Q[int((state[0] + X_MAX) / X_ACCURACY)][
                        int((state[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
                        int((state[2] + THETA_MAX) / THETA_ACCURACY)][
                        int((state[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)]
    av_act = numpy.where(current_state_row >= 0)[1]
    return av_act


def sample_next_action(available_act):
    next_action = int(numpy.random.choice(available_act, 1))
    return next_action



#state x actions matrix
# x, x_dot, theta, theta_dot
# [ 0.0137733  -0.19564042  0.01397023  0.26119658]
# [ 0.00986049 -0.39095898  0.01919416  0.55825294]
# [ 0.00204131 -0.58634504  0.03035922  0.85692077]
# [-0.00968559 -0.78186719  0.04749763  1.15899309]
# [-0.02532293 -0.9775748   0.0706775   1.46618248]
# [-0.04487443 -1.1734874   0.10000114  1.78007959]
# [-0.06834418 -1.36958233  0.13560274  2.10210383]
# [-0.09573582 -1.56578011  0.17764481  2.433444  ]
# [-0.12705143 -1.7619274   0.22631369  2.77498798]
#
# x --> -2.4 to +2.4, interval 0.001? //0.00391281, 0.00781918, 0.0117269
# x_dot --> -1.7619274 to +1.7619274, interval 0.1? //0.19531856, 0.19538606, 0.19552215,
# theta --> -0.22631369 to +0.22631369, interval 0.001? //0.00522393, 0.01116506, 0.01713841
# theta_dot --> -2.77498798 to +2.77498798, interval 0.1? //0.29705636, 0.29866783, 0.3027232
#Q = numpy.matrix(numpy.zeros([MATRIX_SIZE, MATRIX_SIZE]))

#setup
print("Setting up environment...")
LR = 1e-3
env = gym.make('CartPole-v1')
env.reset()

X_ACCURACY = 0.1
X_MAX = 2.4 + X_ACCURACY
X_MIN = -2.4 - X_ACCURACY
X_RANGE = X_MAX - X_MIN
X_MATRIX_SIZE = int(X_RANGE / X_ACCURACY) + 1

X_DOT_ACCURACY = 0.1
X_DOT_MAX = 3.0 + X_DOT_ACCURACY
X_DOT_MIN = -3.0 - X_DOT_ACCURACY
X_DOT_RANGE = X_DOT_MAX - X_DOT_MIN
X_DOT_MATRIX_SIZE = int(X_DOT_RANGE / X_ACCURACY) + 1

THETA_ACCURACY = 0.01
THETA_MAX = 0.3 + THETA_ACCURACY
THETA_MIN = -0.3 - THETA_ACCURACY
THETA_RANGE = THETA_MAX - THETA_MIN
THETA_MATRIX_SIZE = int(THETA_RANGE / THETA_ACCURACY) + 1

THETA_DOT_ACCURACY = 0.1
THETA_DOT_MAX = 3.1 + THETA_DOT_ACCURACY
THETA_DOT_MIN = -3.1 - THETA_DOT_ACCURACY
THETA_DOT_RANGE = THETA_DOT_MAX - THETA_DOT_MIN
THETA_DOT_MATRIX_SIZE = int(THETA_DOT_RANGE / THETA_DOT_ACCURACY) + 1

Q = []
R = []
for i in range(0, X_MATRIX_SIZE):
    #print(i)
    Q.append([])
    R.append([])
    for j in range(0, X_DOT_MATRIX_SIZE):
        Q[i].append([])
        R[i].append([])
        for k in range(0, THETA_MATRIX_SIZE):
            Q[i][j].append([])
            R[i][j].append([])
            for l in range(0, THETA_DOT_MATRIX_SIZE):
                Q[i][j][k].append([])
                R[i][j][k].append([])
                for m in range(0, 2):
                    Q[i][j][k][l].append(0)
                    R[i][j][k][l].append(0)

goal_steps = 500 #frames of success

#initial samples
print("Creating initial samples...")
initial_games = 1000
Q, R = initial_population(initial_games, Q, R)


#training
print("Training...")
gamma = 0.8
scores = []
for i in range(0, 1000):
    print(i)
    j = numpy.random.normal(X_MATRIX_SIZE/2, X_MATRIX_SIZE/6)
    for j in range(0, X_MATRIX_SIZE):
        for k in range(0, X_DOT_MATRIX_SIZE):
            for l in range(0, THETA_MATRIX_SIZE):
                for m in range(0, THETA_DOT_MATRIX_SIZE):
                    for action in range(0, 2):
                        max_value = max(Q[j][k][l][m][0], Q[j][k][l][m][1])
                        result = R[j][k][l][m][action] + (gamma * max_value)
                        Q[j][k][l][m][action] = result

#    x = numpy.random.normal(0.0, X_MAX/5.0)
#    x_dot = numpy.random.normal(0.0, X_DOT_MAX/5.0)
#    theta = numpy.random.normal(0.0, THETA_MAX/5.0)
#    theta_dot = numpy.random.normal(0.0, THETA_DOT_MAX/5.0)
#    current_state = [x, x_dot, theta, theta_dot]
#    action = numpy.random.choice([0, 1])
#    new_score, Q, R = update(current_state, action, gamma, Q, R)
#    scores.append(new_score)

state = [-2*X_ACCURACY, -2*X_DOT_ACCURACY, -2*THETA_ACCURACY, -2*THETA_DOT_ACCURACY]
for i in range(0, 5):
    print(Q[int((state[0] + X_MAX) / X_ACCURACY)][
        int((state[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
        int((state[2] + THETA_MAX) / THETA_ACCURACY)][
        int((state[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)])
    state = [state[0] + X_ACCURACY, state[1] + X_DOT_ACCURACY, state[2] + THETA_ACCURACY, state[3] + THETA_DOT_ACCURACY]
#print(Q[int(X_MATRIX_SIZE/2)][int(X_DOT_MATRIX_SIZE/2)][int(THETA_MATRIX_SIZE/2)][int(THETA_DOT_MATRIX_SIZE/2)])
exit()

#testing
print("Testing...")
env.reset()
current_state = [0.0, 0.0, 0.0, 0.0]
for _ in range(0, goal_steps):
    env.render()
    action = 0
    if (Q[int((current_state[0] + X_MAX) / X_ACCURACY)][
        int((current_state[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
        int((current_state[2] + THETA_MAX) / THETA_ACCURACY)][
        int((current_state[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][0] <=
        Q[int((current_state[0] + X_MAX) / X_ACCURACY)][
        int((current_state[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
        int((current_state[2] + THETA_MAX) / THETA_ACCURACY)][
        int((current_state[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][1]
    ):
        action = 1
    #observation --> self.state = (x, x_dot, theta, theta_dot)
    current_state, reward, done, info = env.step(action)
    time.sleep(0.1)

    if (done):
        break

exit()