import numpy as np

#SARSA class that has function to train and test simple treasure finding path
class SARSA:

    def __init__(self,a,r,action,reward,Q):
        if a is None:
            self.a = 0.5
        if r is None:
            self.r = 0.75

        self.a = a
        self.r = r
        self.action = action
        self.reward = reward
        self.Q = Q

    def trainer(self,i):
        for j in range(i):
            state = np.random.randint(0,int(len(self.Q)))
            currentActions = []

            for k in range(int(len(self.Q))):
                if 0 <= self.reward[state,k]:
                    currentActions.append(k)

            next = int(np.random.choice(currentActions,1))

            nextActions = []

            for act,val in enumerate(self.Q[next,]):
                if 0 < val:
                    nextActions.append(act)

            nextAction = int(np.random.choice(nextActions,1))
            timeDiff = self.reward[state,next] + self.r * self.Q[next,nextAction] - self.Q[state,next]
            self.Q[state,next] += self.a * timeDiff

    def tester(self,start,end):
        state = start
        path = [state]

        while state != end:
            nextActions = []

            for act,val in enumerate(self.Q[state,]):
                if 0 < val:
                    nextActions.append(act)

            next = int(np.random.choice(nextActions,1))
            path.append(next)
            state = next

        return path

#Q-Learning class that has functions to train and test simple treasure finding path
class QL:

    def __init__(self, a, r, action, reward, Q):
        if a is None:
            self.a = 0.5
        if r is None:
            self.r = 0.75

        self.a = a
        self.r = r
        self.action = action
        self.reward = reward
        self.Q = Q

    def trainer(self, i):
        for j in range(i):
            state = np.random.randint(0, int(len(self.Q)))
            currentActions = []

            for k in range(int(len(self.Q))):
                if 0 <= self.reward[state,k]:
                    currentActions.append(k)

            nextState = int(np.random.choice(currentActions,1))
            timeDiff = self.reward[state,nextState] + self.r * self.Q[nextState,np.argmax(self.Q[nextState,])] - self.Q[state,nextState]
            self.Q[state,nextState] += self.a * timeDiff

    def tester(self,start,end):
        state = start
        path = [state]

        while state != end:
            next = np.argmax(self.Q[state,])
            path.append(next)
            state = next

        return path
