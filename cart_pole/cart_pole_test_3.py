import gym
import numpy
import time
from statistics import mean

print("Setting up environment...")
env = gym.make('CartPole-v1')
env.reset()
GAMMA = 0.9
#LR = 0.16
LR = 1.0

X_ACCURACY = 4.8 #4.8
X_MAX = 2.4 #+ X_ACCURACY
X_MIN = -2.4 #- X_ACCURACY
X_RANGE = X_MAX - X_MIN
X_MATRIX_SIZE = int(X_RANGE / X_ACCURACY) #+ 1

X_DOT_ACCURACY = 3.6 #3.6
X_DOT_MAX = 3.6 #+ X_DOT_ACCURACY
X_DOT_MIN = -3.6 #- X_DOT_ACCURACY
X_DOT_RANGE = X_DOT_MAX - X_DOT_MIN
X_DOT_MATRIX_SIZE = int(X_DOT_RANGE / X_DOT_ACCURACY) #+ 1

THETA_ACCURACY = 0.1 #0.04 #0.065
THETA_MAX = 0.3 + THETA_ACCURACY
THETA_MIN = -0.3 - THETA_ACCURACY
THETA_RANGE = THETA_MAX - THETA_MIN
THETA_MATRIX_SIZE = int(THETA_RANGE / THETA_ACCURACY) #+ 1

THETA_DOT_ACCURACY = 0.3 #0.3 #0.382
THETA_DOT_MAX = 3.7 + THETA_DOT_ACCURACY
THETA_DOT_MIN = -3.7 - THETA_DOT_ACCURACY
THETA_DOT_RANGE = THETA_DOT_MAX - THETA_DOT_MIN
THETA_DOT_MATRIX_SIZE = int(THETA_DOT_RANGE / THETA_DOT_ACCURACY) #+ 1

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

print("Training...")
start_time = time.process_time()
for i in range(0, 3000): #episodes 500
    env.reset()
    action = numpy.random.choice([0, 1])
    prev_observation = None
    LR *= 0.999 #0.999
    if (LR < 0.16):
        LR = 0.16
    while (True):
    #for j in range(0, 1000): #goal frames
        observation, reward, done, info = env.step(action)
        if (not prev_observation is None):
            Q[int((prev_observation[0] + X_MAX) / X_ACCURACY)][
                int((prev_observation[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
                int((prev_observation[2] + THETA_MAX) / THETA_ACCURACY)][
                int((prev_observation[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][
                action] = \
            ((1.0 - LR) * Q[int((prev_observation[0] + X_MAX) / X_ACCURACY)][
                int((prev_observation[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
                int((prev_observation[2] + THETA_MAX) / THETA_ACCURACY)][
                int((prev_observation[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][
                action]) + \
            (LR * (reward + (GAMMA * max(
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
print("\t" + str(end_time - start_time) + " sec")

# print("Testing...")
# for _ in range(0, 100):
#     env.reset()
#     current_state = [0.0, 0.0, 0.0, 0.0]
#     score_avg = 0.0
#     for i in range(0, 100):
#         #time.sleep(1)
#         score = 0
#         while (True):
#         #for _ in range(0, 1000): #goal frames
#             #env.render()
#             action = 0
#             if (Q[int((current_state[0] + X_MAX) / X_ACCURACY)][
#                 int((current_state[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
#                 int((current_state[2] + THETA_MAX) / THETA_ACCURACY)][
#                 int((current_state[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][0] <=
#                 Q[int((current_state[0] + X_MAX) / X_ACCURACY)][
#                 int((current_state[1] + X_DOT_MAX) / X_DOT_ACCURACY)][
#                 int((current_state[2] + THETA_MAX) / THETA_ACCURACY)][
#                 int((current_state[3] + THETA_DOT_MAX) / THETA_DOT_ACCURACY)][1]
#             ):
#                 action = 1
#             current_state, reward, done, info = env.step(action)
#             score += reward
#             #time.sleep(0.1)

#             if (done):
#                 #print("Score: " + str(score))
#                 #time.sleep(10)
#                 if (score < 200):
#                     print(score)
#                     #exit()
#                 env.reset()
#                 current_state = [0.0, 0.0, 0.0, 0.0]
#                 score_avg = ((score_avg * i) + score) / (i+1)
#                 #print("\t" + str(score_avg))
#                 break

#     if (score_avg < 200):
#         print("100-trial Mean: " + str(round(score_avg, 3)))

print("Showing...")
for _ in range(0, 10):
    env.reset()
    current_state = [0.0, 0.0, 0.0, 0.0]
    while (True):
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
        current_state, reward, done, info = env.step(action)

        if (done):
            break
