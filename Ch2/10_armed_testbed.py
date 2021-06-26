import numpy as np
import random
import matplotlib as mpt
from matplotlib import pyplot as plt
from statistics import mean

BANDIT_SIZE = 10
TIME_STEPS  = 1000
SAMPLE      = 2000

class Bandit:
    def __init__(self, epsilon = 0, initial = 0):
        # @q      : Real reward value of actions
        # @epsilon: probability for exploration
        # @qt     : Estimated reward value of action till a particular time step 
        # format [qt(a), no of times selected]
        # @Reward : list to store the reward
        gauss_sample = np.random.normal(size = BANDIT_SIZE)
        
        self.q = {a: gauss_sample[a - 1] for a in range(1, BANDIT_SIZE + 1)}
        self.epsilon = epsilon
        self.initial = initial
        self.qt = {a: (initial, 0) for a in range(1, BANDIT_SIZE + 1)}
        self.reward = []
        self.best_action_taken = []
        self.best_action = max(self.q, key=self.q.get)

    def step(self):
        if random.random() < self.epsilon:
            #Exploring other actions
            action = random.randint(1, BANDIT_SIZE)
        else:
            #Greedy approach
            action = max(self.qt, key=self.qt.get)

        reward_obtained = self.getRewardValue(action)

        self.reward.append(reward_obtained)
        if action != self.best_action:
            self.best_action_taken.append(False)
        else:
            self.best_action_taken.append(True)

        self.updateAverage(action, reward_obtained)

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


def avg_reward_graph(tasks_1, tasks_2, tasks_3):
    plot_reward_list_1 = [mean([task.reward[i] for task in tasks_1]) for i in range(TIME_STEPS)]
    plot_reward_list_2 = [mean([task.reward[i] for task in tasks_2]) for i in range(TIME_STEPS)]
    plot_reward_list_3 = [mean([task.reward[i] for task in tasks_3]) for i in range(TIME_STEPS)]

    plt.subplot(1, 2, 1)
    plt.plot(plot_reward_list_1, label=f'epsilon = {tasks_1[0].epsilon}')
    plt.plot(plot_reward_list_2, label=f'epsilon = {tasks_2[0].epsilon}')
    plt.plot(plot_reward_list_3, label=f'epsilon = {tasks_3[0].epsilon}')

    plt.title('Average Reward through learning time steps')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

def optimal_action_graph(tasks_1, tasks_2, tasks_3):
    plot_best_action_1 = []
    plot_best_action_2 = []
    plot_best_action_3 = []

    for i in range(TIME_STEPS):
        n1 = 0
        for task in tasks_1:
            if task.best_action_taken[i]:
                n1 += 1

        plot_best_action_1.append((n1/SAMPLE)*100)

        n2 = 0
        for task in tasks_2:
            if task.best_action_taken[i]:
                n2 += 1

        plot_best_action_2.append((n2/SAMPLE)*100)

        n3 = 0
        for task in tasks_3:
            if task.best_action_taken[i]:
                n3 += 1

        plot_best_action_3.append((n3/SAMPLE)*100)
    
    plt.subplot(1, 2, 2)
    plt.plot(plot_best_action_1, label=f'epsilon = {tasks_1[0].epsilon}')
    plt.plot(plot_best_action_2, label=f'epsilon = {tasks_2[0].epsilon}')
    plt.plot(plot_best_action_3, label=f'epsilon = {tasks_3[0].epsilon}')

    plt.title('Percentage of actions where best actions is chosen through learning time steps')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
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

    plt.title('Percentage of actions where best actions is chosen through learning time steps')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

def sim(epsilon = 0, initial = 0):
    tasks = [Bandit(epsilon=epsilon, initial=initial) for i in range(SAMPLE)]
    for task in tasks:
        for t in range(TIME_STEPS):
            task.step()

    return tasks

if __name__ == "__main__":

    tasks_e_0_0 = sim()
    tasks_e_0__inf = sim(initial=float("inf"))
    tasks_e_0_inf = sim(initial=float("-inf"))
    tasks_e_0_01_0 = sim(epsilon = 0.01, initial=0)
    tasks_e_0_01__inf = sim(epsilon = 0.01, initial=float("inf"))
    tasks_e_0_01_inf = sim(epsilon = 0.01, initial=float("-inf"))
    tasks_e_0_1_0 = sim(epsilon = 0.1, initial=0)
    tasks_e_0_1__inf = sim(epsilon = 0.1, initial=float("inf"))
    tasks_e_0_1_inf = sim(epsilon = 0.1, initial=float("-inf"))

    # Show graphs
    # plt.figure(figsize=(17, 7))
    # avg_reward_graph(tasks_e_0, tasks_e_0_01, tasks_e_0_1)
    optimal_action_general((tasks_e_0_0, tasks_e_0__inf, tasks_e_0_inf, tasks_e_0_01_0, tasks_e_0_01__inf, tasks_e_0_01_inf, tasks_e_0_1_0, tasks_e_0_1__inf, tasks_e_0_1_inf))
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