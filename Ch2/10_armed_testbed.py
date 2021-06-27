import numpy as np
import random
import matplotlib as mpt
from matplotlib import pyplot as plt
from statistics import mean
from math import e

BANDIT_SIZE = 10
TIME_STEPS  = 5
SAMPLE      = 2

class Bandit:
    def __init__(self, epsilon = 0, initial = 0, step_size = 0.1, baseline = False):
        # @q      : Real reward value of actions
        # @epsilon: probability for exploration
        # @qt     : Estimated reward value of action till a particular time step 
        # format [qt(a), no of times selected]
        # @Reward : list to store the reward
        gauss_sample = np.random.normal(size = BANDIT_SIZE)
        
        self.q = {a: gauss_sample[a - 1] for a in range(1, BANDIT_SIZE + 1)}
        self.qt = {a: (initial, 0) for a in range(1, BANDIT_SIZE + 1)}

        self.epsilon = epsilon
        self.initial = initial
        self.step_size = step_size
        self.gradient_baseline = baseline

        self.avg_reward = 0
        self.time_step = 0

        self.reward = []

        self.best_action_taken = []
        self.best_action = max(self.q, key=self.q.get)

        self.preferences = {a: 0 for a in range(1, BANDIT_SIZE + 1)}

    def e_greedy(self):
        if random.random() < self.epsilon:
            #Exploring other actions
            action = random.randint(1, BANDIT_SIZE)
        else:
            #Greedy approach
            action = max(self.qt, key=self.qt.get)

        return action

    def gradientPref(self):
        return

    def step(self):
        action = self.e_greedy()
        reward_obtained = self.getRewardValue(action)

        self.reward.append(reward_obtained)
        if action != self.best_action:
            self.best_action_taken.append(False)
        else:
            self.best_action_taken.append(True)

        self.updateEstimate(action, reward_obtained)

        self.gradientUpdate(action, reward_obtained)

        self.time_step += 1
        self.updateAvgReward(reward_obtained)

    def getRewardValue(self, a):
        return self.q[a] + np.random.normal()

    def updateAvgReward(self, reward):
        self.avg_reward += (1/self.time_step)*(reward - self.avg_reward)

    def _calc_prob(self, action_index):
        prob_t = np.exp(self.preferences[action_index])/sum([np.exp(val) for val in self.preferences.values()])
        return prob_t

    def gradientUpdate(self, index, reward):
        for action in self.preferences:
            var = 1 if action == index else 0
            if self.gradient_baseline:
                baseline = self.avg_reward
            else:
                baseline = 0

            self.preferences[action] += self.step_size*(var - self._calc_prob(action))*(reward - baseline)
    
    def sampleAverages(self, index, reward):
        qk, n = self.qt[index]

        if n == 0:
            qk = reward
            n += 1
        else:
            step_size = 1/n
            qk += ((reward - qk) * step_size)
            n += 1

        self.qt[index] = (qk, n)

    def updateEstimate(self, index, reward):
        self.sampleAverages(index, reward)

    def getState(self):
        return (self.q, self.qt)
        # return {i:self._calc_prob(i) for i in self.preferences}


def avg_reward_general(graphs):
    for lst in graphs:
        plot_reward_list = [mean([task.reward[i] for task in lst]) for i in range(TIME_STEPS)]
        plt.plot(plot_reward_list, label=f'epsilon = {lst[0].epsilon}, initial = {lst[0].initial}')

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

def optimal_action_general(graphs):
    for lst in graphs:
        plot_best_action = []
        for i in range(TIME_STEPS):
            n = 0
            for task in lst:
                if task.best_action_taken[i]:
                    n += 1
            plot_best_action.append((n/SAMPLE)*100)
        
        plt.plot(plot_best_action, label=f'epsilon = {lst[0].epsilon}, initial = {lst[0].initial}')
    
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

# Fig 2.1 graph
def avg_reward_graph(tasks_1, tasks_2, tasks_3):
    plt.subplot(1, 2, 1)
    avg_reward_general((tasks_1, tasks_2, tasks_3))

    plt.title('Average Reward through learning time steps')

#Fig 2.1 graph
def optimal_action_graph(tasks_1, tasks_2, tasks_3):
    plt.subplot(1, 2, 2)
    optimal_action_general((tasks_1, tasks_2, tasks_3))

    plt.title('Percentage of actions where best actions is chosen through learning time steps')

def sim(eps = 0, init = 0):
    tasks = [Bandit(epsilon=eps, initial=init) for i in range(SAMPLE)]
    for task in tasks:
        for t in range(TIME_STEPS):
            task.step()
            print(task.getState())

    return tasks

if __name__ == "__main__":
    # SIMULATE LEARNING

    tasks_e_0 = sim()
    tasks_e_0_01 = sim(eps = 0.01)
    tasks_e_0_1 = sim(eps = 0.1)

    # tasks_e_0_0 = sim()
    # tasks_e_0__inf = sim(init=float("inf"))
    # tasks_e_0_inf = sim(init=float("-inf"))
    # tasks_e_0_01_0 = sim(eps = 0.01, init=0)
    # tasks_e_0_01__inf = sim(eps = 0.01, init=float("inf"))
    # tasks_e_0_01_inf = sim(eps = 0.01, init=float("-inf"))
    # tasks_e_0_1_0 = sim(eps = 0.1, init=0)
    # tasks_e_0_1__inf = sim(eps = 0.1, init=float("inf"))
    # tasks_e_0_1_inf = sim(eps = 0.1, init=float("-inf"))

    # task_e_0_5 = sim(init=5)
    # tasks_e_0_1_0 = sim(eps=0.1)
    # tasks_e_0_10 = sim(init=10)
    # tasks_e_0_20 = sim(init=20)
    # task_e_0__inf = sim(init=float('inf'))

    # SHOW GRAPHS
    # plt.figure(figsize=(17, 7))

    avg_reward_graph(tasks_e_0, tasks_e_0_1, tasks_e_0_01)
    optimal_action_graph(tasks_e_0, tasks_e_0_01, tasks_e_0_1)
    # optimal_action_general((tasks_e_0_1_0, task_e_0_5))
    # optimal_action_general((task_e_0_5, tasks_e_0_1_0, task_e_0__inf, tasks_e_0_10, tasks_e_0_20))

    plt.show()

    # TESTING
    # for i, bandit_task in enumerate(tasks_e_0):
    #     q, qt = bandit_task.getState()
    #     print(f"{i}:\n{q}\n{qt}")

    # for i, bandit_task in enumerate(tasks_e_0_01):
    #     q, qt = bandit_task.getState()
    #     print(f"{i}:\n{q}\n{qt}")

    # for i, bandit_task in enumerate(tasks_e_0_1):
    #     q, qt = bandit_task.getState()
    #     print(f"{i}:\n{q}\n{qt}")