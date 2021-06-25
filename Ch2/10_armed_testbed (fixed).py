import numpy as np
import random
import matplotlib.pyplot as plt

BANDIT_SIZE = 10
TIME_STEPS  = 1000
SAMPLE      = 2000

class Bandit:
    def __init__(self, epsilon = 0):
        # @q      : Real reward value of actions
        # @epsilon: probability for exploration
        # @qt     : Reward value of action till a particular step 
        # format [qt(a), no of times selected]
        # @Reward : list to store the rewards
        gauss_sample = np.random.normal(size = BANDIT_SIZE)
        
        self.q = {a: gauss_sample[a - 1] + 100 for a in range(0, BANDIT_SIZE)}
        self.epsilon = epsilon
        self.qt = [(-float('inf'),0) for i in range(BANDIT_SIZE)]
        self.reward = []

    def step(self):
        if random.random()<self.epsilon: #Exploring other actions
            indice_to_be_explored = random.randint(0, BANDIT_SIZE-1)
            reward_obtained = self.getRewardValue(indice_to_be_explored)
            self.reward.append(reward_obtained)
            self.update_average(indice_to_be_explored, reward_obtained)

        else:
            #Greedy approach
            max_element =  max(self.qt)
            indice_having_maxqt = self.qt.index(max_element)
            reward_obtained = self.getRewardValue(indice_having_maxqt)
            self.reward.append(reward_obtained)
            self.update_average(indice_having_maxqt, reward_obtained)      

    def getRewardValue(self, a):
        return self.q[a] + np.random.normal(scale=0.1)

    def update_average(self, index, reward):
        a,b = self.qt[index]
        if b==0:
            a = reward
            b = b + 1

        else:
            a = ((a*b)+reward)/(b+1)
            b = b + 1

        self.qt[index] = (a,b)

    def tellstate(self):
        print(self.q, self.qt)
def average(lis):
    return (sum(lis)/len(lis))
def plot():
    plt.xlabel('steps')
    plt.ylabel('reward')

    temp1,temp2,temp3=[],[],[]
    for t in range(TIME_STEPS):
        temp1.append(average([x.reward[t] for x in list_e_0 ] ))
        temp2.append(average([x.reward[t] for x in list_e_0_1 ] ))
        temp3.append(average([x.reward[t] for x in list_e_0_01 ] ))
    plt.plot(range(1,TIME_STEPS+1),temp1)
    plt.plot(range(1,TIME_STEPS+1),temp2)
    plt.plot(range(1,TIME_STEPS+1),temp3)

    plt.show()    
if __name__ == "__main__":

    list_e_0 = [Bandit() for i in range(SAMPLE)]
    for x in list_e_0:
        for t in range(TIME_STEPS):
            x.step()
    print(1)
    list_e_0_01 = [Bandit(epsilon=0.01) for i in range(SAMPLE)]
    for x in list_e_0_01:
        for t in range(TIME_STEPS):
            x.step()
    print(1)
    list_e_0_1 = [Bandit(epsilon=0.1) for i in range(SAMPLE)]
    for y in list_e_0_1:
        for t in range(TIME_STEPS):
            y.step()
    print(1)
    plot()
