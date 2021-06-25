import numpy as np

N = 10
SAMPLE = 2000

class Task:
    def __init__(self):
        gauss_sample = np.random.normal(size = N)
        self.q = {i: gauss_sample[i - 1] for i in range(1, N+1)}

    def getActionValue(self, a):
        return self.q[a]

    def getRewardValue(self, a):
        return self.q[a] + np.random.normal()
        
def Train(task):
    
bandits = [Task() for i in range(2000)]

# Testing
for i in range(0, 100):
    print(f"{i}: ", ', '.join(map(str, [bandits[i].getActionValue(j) for j in range(1, N+1)])))