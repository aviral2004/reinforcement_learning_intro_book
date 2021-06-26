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
        self.qt = {a: (0, 0) for a in range(1, BANDIT_SIZE + 1)}
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
    n_best_action_taken_1 = 0
    n_best_action_taken_2 = 0
    n_best_action_taken_3 = 0

    for i in range(1, TIME_STEPS + 1):
        for task in tasks_1:
            if task.best_action_taken[i - 1]:
                n_best_action_taken_1 += 1
        plot_best_action_1.append((n_best_action_taken_1/(i*SAMPLE))*100)

        for task in tasks_2:
            if task.best_action_taken[i - 1]:
                n_best_action_taken_2 += 1
        plot_best_action_2.append((n_best_action_taken_2/(i*SAMPLE))*100)

        for task in tasks_3:
            if task.best_action_taken[i - 1]:
                n_best_action_taken_3 += 1
        plot_best_action_3.append((n_best_action_taken_3/(i*SAMPLE))*100)
    
    plt.subplot(1, 2, 2)
    plt.plot(plot_best_action_1, label=f'epsilon = {tasks_1[0].epsilon}')
    plt.plot(plot_best_action_2, label=f'epsilon = {tasks_2[0].epsilon}')
    plt.plot(plot_best_action_3, label=f'epsilon = {tasks_3[0].epsilon}')

    plt.title('Percentage of actions where best actions is chosen through learning time steps')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

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

    # Show graphs
    plt.figure(figsize=(17, 7))
    avg_reward_graph(tasks_e_0, tasks_e_0_01, tasks_e_0_1)
    optimal_action_graph(tasks_e_0, tasks_e_0_01, tasks_e_0_1)
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