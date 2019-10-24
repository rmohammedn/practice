import numpy as np

class Simulator:
    """ ... """

    def __init__(self, max_r=10, min_r=2, length=5):
        """ initialising everthing """

        self.max_r = max_r
        self.min_r = min_r
        self.length = length - 1
        self.state = 0

    def nextState(self, action):
        """ return next state of agent for given action """
        
        if action == 1 and self.state < self.length - 1:
            self.state += 1
            reward = 0
        elif action == 1 and self.state == self.length - 1:
            self.state += 1
            reward = self.max_r
        elif action == 1 and self.state == self.length:
            reward = 0
        elif action == 0 and self.state == 0:
            reward = 0
        else:
            self.state = 0
            reward = self.min_r
        return self.state, reward

    def reset(self):
        """ reset the state """
        self.state = 0
        return self.state

class ClassicQAgent:
    """..."""

    def __init__(self, state_size=5, action_size=2, discount=1.0, learning_rate=1, expo_rate=1.0, iteration=10000):
        """.............."""

        self.q_table = np.zeros((state_size, action_size))
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.expo_rate = expo_rate
        self.discount = discount
        self.expo_delta = self.expo_rate/self.iteration

    def nextAction(self, state):
        """ ..........."""

        if np.random.random() > self.expo_rate:
            return self.greedyAction(state)
        else:
            return self.randomAction()

    def greedyAction(self, state):
        """ ...... """

        if self.q_table[state, 0] > self.q_table[state, 1]:
            return 0
        elif self.q_table[state, 0] < self.q_table[state, 1]:
            return 1
        else:
            return self.randomAction()

    def randomAction(self):
        """ ....."""

        if np.random.random() > 0.5:
            return 1
        else:
            return 0

    def train(self, old_state, action, new_state, reward):
        """ ...... """

        old_q = self.q_table[old_state]
        new_q = self.q_table[new_state]
        #print(self.q_table)
        #print(old_state, action, reward)

        q_update = old_q[action] + self.learning_rate * (reward + self.discount * np.amax(new_q) - old_q[action])
        self.q_table[old_state, action] = q_update

        if self.expo_rate > 0:
            self.expo_rate -= self.expo_delta

def main():
    """ ...."""

    arena = Simulator()
    agent = ClassicQAgent()
    arena.reset()

    for i in range(agent.iteration):
        state = arena.state
        next_action = agent.nextAction(state)
        new_state, reward = arena.nextState(next_action)
        agent.train(state, next_action, new_state, reward)

    print(agent.q_table)

if __name__ == "__main__":
    main()
