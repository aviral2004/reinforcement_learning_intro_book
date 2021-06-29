import numpy as np
import random
from matplotlib import pyplot as plt
from statistics import mean
from math import e
from numpy.core.defchararray import greater
from numpy.lib.function_base import average
from tqdm import tqdm

import cProfile

BANDIT_SIZE = 10
TIME_STEPS  = 1000
SAMPLE      = 2000

class Bandit:
    def __init__(self, epsilon = 0, initial = 0, step_size = 0.1, baseline = False, mean = 0, gradient = False, sample_averages = False):
        # @q      : Real reward value of actions
        # @epsilon: probability for exploration
        # @qt     : Estimated reward value of action till a particular time step 
        # format [qt(a), no of times selected]
        # @Reward : list to store the reward
        gauss_sample = np.random.normal(loc=mean, size = BANDIT_SIZE)
        
        self.q = {a: gauss_sample[a - 1] for a in range(1, BANDIT_SIZE + 1)}
        self.qt = {a: (initial, 0) for a in range(1, BANDIT_SIZE + 1)}

        self.epsilon = epsilon
        self.initial = initial
        if gradient:
            self.step_size = step_size
        elif sample_averages:
            self.step_size = lambda x: 1/x
        else:
            self.step_size = lambda x: step_size
        
        self.gradient_baseline = baseline
        self.gradient = gradient

        self.avg_reward = 0
        self.time_step = 0

        self.reward = []

        self.best_action_taken = []
        self.best_action = max(self.q, key=self.q.get)

        # H_t
        self.preferences = np.zeros(BANDIT_SIZE)

    def e_greedy(self):
        if random.random() < self.epsilon:
            #Exploring other actions
            action = random.randint(1, BANDIT_SIZE)
        else:
            #Greedy approach
            action = max(self.qt, key=self.qt.get)

        return action

    def gradientPref(self):
        return random.choices(np.arange(BANDIT_SIZE), weights=self._calc_prob(self.preferences), k = 1)[0] + 1

    def step(self):
        if self.gradient:
            action = self.gradientPref()
        else:
            action = self.e_greedy()
        reward_obtained = self.getRewardValue(action)

        # For documenting purposes
        self.reward.append(reward_obtained)
        if action != self.best_action:
            self.best_action_taken.append(False)
        else:
            self.best_action_taken.append(True)

        # updates either qt or ht
        self.updateEstimate(action, reward_obtained)

        self.time_step += 1
        self.updateAvgReward(reward_obtained)

    def getRewardValue(self, a):
        return self.q[a] + np.random.normal()

    def updateAvgReward(self, reward):
        self.avg_reward += (1/self.time_step)*(reward - self.avg_reward)

    def _calc_prob(self, action_val):
        prob_t = np.exp(action_val)/(np.exp(self.preferences)).sum()
        return prob_t

    def gradientUpdate(self, index, reward):
        one_zero = np.zeros(BANDIT_SIZE)
        one_zero[index - 1] = 1

        if self.gradient_baseline:
            baseline = self.avg_reward
        else:
            baseline = 0

        pi_t = self._calc_prob(self.preferences)

        # reward = R_t
        # baseline = R'_t
        self.preferences += self.step_size*(one_zero - pi_t)*(reward - baseline)
    
    def sampleAverages(self, index, reward):
        qk, n = self.qt[index]

        if n == 0:
            qk = reward
            n += 1
        else:
            qk += ((reward - qk) * self.step_size(n))
            n += 1

        self.qt[index] = (qk, n)

    def updateEstimate(self, index, reward):
        if self.gradient:
            self.gradientUpdate(index, reward)
        else:
            self.sampleAverages(index, reward)

    def getState(self):
        return (self.q, self.qt)


def avg_reward_general(graphs):
    for lst in graphs:
        plot_reward_list = [mean([task.reward[i] for task in lst]) for i in range(TIME_STEPS)]
        plt.plot(plot_reward_list, label=f'epsilon = {lst[0].epsilon}, initial = {lst[0].initial}')

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

def optimal_action_general(graphs):
    for lst in graphs:
        plot_best_action = np.average(np.array([task.best_action_taken for task in lst]), axis=0)*100
        if lst[0].gradient:
            plt.plot(plot_best_action, label=f'step_size = {lst[0].step_size}, baseline = {lst[0].gradient_baseline}')
        else:
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

def performance_graph():
    labels = ['epsilon-greedy', 'gradient bandit', 'optimistic initialisation']

    generators = np.array([
        lambda epsilon: np.average([np.average(task.reward) for task in sim(eps=epsilon, sample_averages=True)]),
        lambda alpha: np.average([np.average(task.reward) for task in sim(gradient=True, step_size=alpha, baseline=True)]),
        lambda initial: np.average([np.average(task.reward) for task in sim(eps=0, init=initial, step_size=0.1)])
    ])

    parameters = np.array([
        np.arange(-7, -1, dtype=float),
        np.arange(-5, 2, dtype=float),
        np.arange(-2, 3, dtype=float)
    ])
    
    # greedy, gradient, greedy_optimistic = [list(map(generators[i], np.power(2, parameters[i]))) for i in range(len(generators))]
    plot_values = [list(map(generators[i], np.power(2, parameters[i]))) for i in range(len(generators))]
    for i, (plot_value, parameter) in enumerate(zip(plot_values, parameters)):
        plt.plot(parameter, plot_value, label=labels[i])

    plt.xlabel(r'$ \alpha / c / Q_0$' +  '\n' +  r'($ Parameter = 2^x $)')
    plt.ylabel('Average Reward over first 1000 steps')
    plt.legend()


def sim(eps = 0, init = 0, step_size = 0.1, baseline = False, mean = 0, gradient = False, sample_averages = False):
    tasks = [Bandit(epsilon=eps, initial=init, step_size=step_size, baseline=baseline, mean=mean, gradient=gradient, sample_averages=sample_averages) for i in range(SAMPLE)]
    for i in tqdm(range(len(tasks))):
        for t in range((TIME_STEPS)):
            tasks[i].step()

    return tasks

if __name__ == "__main__":
    # SIMULATE LEARNING

    # Fig 2.1
    # tasks_e_0 = sim(sample_averages = True)
    # tasks_e_0_01 = sim(sample_averages = True, eps = 0.01)
    # tasks_e_0_1 = sim(sample_averages = True, eps = 0.1)

    # tasks_e_0_0 = sim()
    # tasks_e_0__inf = sim(init=float("inf"))
    # tasks_e_0_inf = sim(init=float("-inf"))
    # tasks_e_0_01_0 = sim(eps = 0.01, init=0)
    # tasks_e_0_01__inf = sim(eps = 0.01, init=float("inf"))
    # tasks_e_0_01_inf = sim(eps = 0.01, init=float("-inf"))
    # tasks_e_0_1_0 = sim(eps = 0.1, init=0)
    # tasks_e_0_1__inf = sim(eps = 0.1, init=float("inf"))
    # tasks_e_0_1_inf = sim(eps = 0.1, init=float("-inf"))

    # Fig 2.2
    # task_e_0_5 = sim(init=5)
    # tasks_e_0_1_0 = sim(eps=0.1)

    # Fig 2.4
    # task_a_0_1_base = sim(step_size=0.1, baseline=True, mean = 4, gradient = True)
    # task_a_0_1 = sim(step_size=0.1, mean = 4, gradient = True)
    # task_a_0_4_base = sim(step_size=0.4, baseline=True, mean = 4, gradient = True)
    # task_a_0_4 = sim(step_size=0.4, mean = 4, gradient = True)

    # SHOW GRAPHS
    # plt.figure(figsize=(17, 7))

    # Fig 2.1
    # avg_reward_graph(tasks_e_0, tasks_e_0_1, tasks_e_0_01)
    # optimal_action_graph(tasks_e_0, tasks_e_0_01, tasks_e_0_1)

    # Fig 2.2
    # optimal_action_general((task_e_0_5, tasks_e_0_1_0))

    # Fig 2.4
    # optimal_action_general((task_a_0_1_base, task_a_0_1, task_a_0_4_base, task_a_0_4))

    performance_graph()

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