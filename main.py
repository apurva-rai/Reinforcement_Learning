import RL
import numpy as np

if __name__ == '__main__':
    a = 1
    r = 0.8
    action = [0,1,2,3,4,5,6,7,8]
    reward = np.array([[-1,0,-1,0,-1,-1,-1,-1,-1],
                        [-1,-1,0,-1,0,-1,-1,-1,-1],
                        [-1,-1,-1,-1,-1,0,-1,-1,-1],
                        [-1,-1,-1,-1,0,-1,0,-1,-1],
                        [-1,-1,0,-1,-1,0,0,0,100],
                        [-1,-1,-1,-1,-1,-1,-1,-1,100],
                        [-1,-1,-1,-1,-1,-1,-1,0,-1],
                        [-1,-1,-1,-1,-1,-1,-1,-1,100],
                        [-1,-1,-1,-1,-1,-1,-1,-1,100]])

    Q = np.array(np.zeros([9,9]))

    start = 0
    end = 8

    print("Maze matrix\n", reward)

    print("\nQ-learning results:")
    QL_agent = RL.QL(a, r, action, reward, Q)
    QL_agent.trainer(i=1000)
    print("Updated state matrix\n", QL_agent.Q)
    print("Test path with start state", start, "and end state", end, ":", QL_agent.tester(start, end))

    print("\nSARSA results:")
    SARSA_agent = RL.SARSA(a, r, action, reward, Q)
    SARSA_agent.trainer(i=1000)
    print("Updated state matrix\n", SARSA_agent.Q)
    print("Test path with start state", start, "and end state", end, ":", SARSA_agent.tester(start, end))
