import numpy as np
import random
import matplotlib as mpt
from matplotlib import pyplot as plt
from statistics import mean


BANDIT_SIZE = 10
TIME_STEPS  = 1000
SAMPLE      = 2000

class Bandit:
    def __init__(self, epsilon = 0):
        # @q      : Real reward value of actions
        # @epsilon: probability for exploration
        # @qt     : Estimated reward value of action till a particular time step 
        # format [qt(a), no of times selected]
        # @Reward : list to store the reward
        gauss_sample = np.random.normal(size = BANDIT_SIZE)
        
        self.q = {a: gauss_sample[a - 1] for a in range(1, BANDIT_SIZE + 1)}
        self.epsilon = epsilon
        self.qt = {a: (float('-inf'),0) for a in range(1, BANDIT_SIZE + 1)}
        self.reward = []

    def step(self):
        if random.random() < self.epsilon: #Exploring other actions
            curr_action = random.randint(1, BANDIT_SIZE)
            reward_obtained = self.getRewardValue(curr_action)
            self.reward.append(reward_obtained)
            self.updateAverage(curr_action, reward_obtained)
        else:
            #Greedy approach
            max_reward_action = max(self.qt, key=self.qt.get)
            reward_obtained = self.getRewardValue(max_reward_action)
            self.reward.append(reward_obtained)
            self.updateAverage(max_reward_action, reward_obtained)      

    def getRewardValue(self, a):
        return self.q[a] + np.random.normal()

    def updateAverage(self, index, reward):
        avg, n = self.qt[index]

        if n == 0:
            avg = reward
            n += 1
        else:
            avg = ((avg*n)+reward)/(n+1)
            n += 1

        self.qt[index] = (avg, n)

    def getState(self):
        return (self.q, self.qt)

if __name__ == "__main__":

    tasks_e_0 = [Bandit() for i in range(SAMPLE)]
    for bandit_task in tasks_e_0:
        for t in range(TIME_STEPS):
            bandit_task.step()

    tasks_e_0_01 = [Bandit(epsilon=0.01) for i in range(SAMPLE)]
    for bandit_task in tasks_e_0_01:
        for t in range(TIME_STEPS):
            bandit_task.step()

    tasks_e_0_1 = [Bandit(epsilon=0.1) for i in range(SAMPLE)]
    for bandit_task in tasks_e_0_1:
        for t in range(TIME_STEPS):
            bandit_task.step()

    #ITERATE THROUGH BANDITS of list_e_0_1 etc... 
    #AND ACCESS THEIR SELF.REWARDS AND TAKE MEAN TO GET VALUES FOR GRAPH

    plot_reward_list = [mean([task.reward[i] for task in tasks_e_0]) for i in range(TIME_STEPS)]
    plt.plot(plot_reward_list)
    plt.show()

    # Testing
    # for i, bandit_task in enumerate(tasks_e_0):
    #     q, qt = bandit_task.getState()
    #     print(f"{i}:\n{q}\n{qt}")

    # for i, bandit_task in enumerate(tasks_e_0_01):
    #     q, qt = bandit_task.getState()
    #     print(f"{i}:\n{q}\n{qt}")

    # for i, bandit_task in enumerate(tasks_e_0_1):
    #     q, qt = bandit_task.getState()
    #     print(f"{i}:\n{q}\n{qt}")