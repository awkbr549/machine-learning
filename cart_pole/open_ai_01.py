import gym
import random
import numpy
from statistics import mean, median, mode
from collections import Counter

import time

LR = 1e-3
env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500 #frames of success
score_req = 50 #games must meet this score to continue
initial_games = 10000


def some_random_games_first():
    for ep in range(0, 5):
        env.reset()
        for t in range(0, goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if (done):
                time.sleep(1)
                break


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(0, initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(0, goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if (len(prev_observation) > 0):
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            if (done):
                break

        if (score >= score_req):
            accepted_scores.append(score)
            for data in game_memory:
                output = [0, 1]
                if (data[1] == 0):
                    output = [1, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    print("Average accepted score: " + str(mean(accepted_scores)))
    print("Median accepted score: " + str(median(accepted_scores)))
    print(Counter(accepted_scores))
    return training_data


#some_random_games_first()
initial_population()