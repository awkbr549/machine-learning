import gym
import numpy
import time
from statistics import mean


def train_and_test(learning_rate):
    #print("Training...")
    start_time = time.process_time()
    for i in range(0, 100000): #episodes
        env.reset()
        action = numpy.random.choice([0, 1])
        prev_observation = None
        while (True):
        #for j in range(0, 1000): #goal frames
            observation, reward, done, info = env.step(action)
            if (not prev_observation is None):
                Q[int((prev_observation[0] + X_MAX) / X_ACCURACY)][
                    int((prev_observation[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
                    int((prev_observation[2] + THETA_MAX) / THETA_ACCURACY)][
                    int((prev_observation[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][
                    action] = \
                ((1.0 - learning_rate) * Q[int((prev_observation[0] + X_MAX) / X_ACCURACY)][
                    int((prev_observation[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
                    int((prev_observation[2] + THETA_MAX) / THETA_ACCURACY)][
                    int((prev_observation[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][
                    action]) + \
                (learning_rate * (reward + (GAMMA * max(
                    Q[int((observation[0] + X_MAX) / X_ACCURACY)][
                        int((observation[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
                        int((observation[2] + THETA_MAX) / THETA_ACCURACY)][
                        int((observation[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][
                        0],
                    Q[int((observation[0] + X_MAX) / X_ACCURACY)][
                        int((observation[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
                        int((observation[2] + THETA_MAX) / THETA_ACCURACY)][
                        int((observation[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][
                        1]
                ))))
            if (done):
                break
            action = numpy.random.choice([0, 1])
            prev_observation = observation

    end_time = time.process_time()
    #print("\t" + str(end_time - start_time) + " sec")

    #print("Testing...")
    env.reset()
    current_state = [0.0, 0.0, 0.0, 0.0]
    score_avg = 0.0
    for i in range(0, 1000):
        score = 0
        while (True):
        #for _ in range(0, 1000): #goal frames
            #env.render()
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
            current_state, reward, done, info = env.step(action)
            score += reward
            #time.sleep(0.1)

            if (done):
                #print("Score: " + str(score))
                #time.sleep(10)
                env.reset()
                score_avg = ((score_avg * i) + score) / (i+1)
                #print("\t" + str(score_avg))
                break

    return score_avg
    #print("1000-trial Mean: " + str(round(score_avg, 3)))


print("Setting up environment...")
env = gym.make('CartPole-v1')
env.reset()

X_ACCURACY = 0.8 #0.1
X_MAX = 2.4 + X_ACCURACY
X_MIN = -2.4 - X_ACCURACY
X_RANGE = X_MAX - X_MIN
X_MATRIX_SIZE = int(X_RANGE / X_ACCURACY) + 1

X_DOT_ACCURACY = 0.8 #0.1
X_DOT_MAX = 3.6 + X_DOT_ACCURACY
X_DOT_MIN = -3.6 - X_DOT_ACCURACY
X_DOT_RANGE = X_DOT_MAX - X_DOT_MIN
X_DOT_MATRIX_SIZE = int(X_DOT_RANGE / X_DOT_ACCURACY) + 1

THETA_ACCURACY = 0.01 #0.01
THETA_MAX = 0.3 + THETA_ACCURACY
THETA_MIN = -0.3 - THETA_ACCURACY
THETA_RANGE = THETA_MAX - THETA_MIN
THETA_MATRIX_SIZE = int(THETA_RANGE / THETA_ACCURACY) + 1

THETA_DOT_ACCURACY = 0.1 #0.1
THETA_DOT_MAX = 3.7 + THETA_DOT_ACCURACY
THETA_DOT_MIN = -3.7 - THETA_DOT_ACCURACY
THETA_DOT_RANGE = THETA_DOT_MAX - THETA_DOT_MIN
THETA_DOT_MATRIX_SIZE = int(THETA_DOT_RANGE / THETA_DOT_ACCURACY) + 1

Q = []
for i in range(0, X_MATRIX_SIZE):
    #print(i)
    Q.append([])
    for j in range(0, X_DOT_MATRIX_SIZE):
        Q[i].append([])
        for k in range(0, THETA_MATRIX_SIZE):
            Q[i][j].append([])
            for l in range(0, THETA_DOT_MATRIX_SIZE):
                Q[i][j][k].append([])
                for m in range(0, 2):
                    Q[i][j][k][l].append(0)

GAMMA = 0.9
LR_DELTA = 0.0001
SCORE_THRESHOLD = 1.0
learning_rate = 0.16 #0.15
#while (True):
for _ in range(0, 1000):
    new_learning_rate = learning_rate
    improvement = 0.0
    score_1 = train_and_test(learning_rate - LR_DELTA)
    score_2 = train_and_test(learning_rate)
    score_3 = train_and_test(learning_rate + LR_DELTA)
    if (score_1 > score_2):
        new_learning_rate = learning_rate - LR_DELTA
        improvement = score_1 - score_2
    elif (score_3 > score_2):
        new_learning_rate = learning_rate + LR_DELTA
        improvement = score_3 - score_2
    #if (learning_rate == new_learning_rate or improvement <= SCORE_THRESHOLD):
    #    break
    learning_rate = round(new_learning_rate, 4)
    print(learning_rate)
    #break

print(learning_rate)

exit()
