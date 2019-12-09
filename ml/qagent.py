import numpy as np
import random
from collections import deque


class QAgent:
    """ Agent will make policy and differnt things """
    def __init__(self, action_size, model, epsilon=1.0, discount=0.8, iteration=2000):

        """ initialize agent parameter """
        self.model = model
        self.epsilon = epsilon
        self.action = action_size
        self.discount = discount
        self.memory = deque(maxlen=1000)
        self.iteration = iteration

    def remember(self, state, action, new_state, reward):
        """ save the collection for training """
        self.memory.append((state, action, new_state, reward))

    def getNextAction(self, state):
        """ it will return next state depending on policy """

        if np.random.random() > self.epsilon:
            return self.greedyPolicy(state)
        else:
            return self.randomAction()

    def greedyPolicy(self, state):
        """ it will take the next state with highest Q value """

        return np.argmax(self.model.getQValue(state))

    def randomAction(self):
        """ ... """
        return np.random.choice(self.action, 1)[0]

    def train(self, batch_size):
        """ will train the model """

        training_input = None
        target = None
        minibatch = random.sample(self.memory, batch_size)
        #print(old_state.shape, new_state.shape)
        for old_state, action, new_state, reward in minibatch:
            old_Q = self.model.getQValue(old_state)
            new_Q = self.model.getQValue(new_state)

            old_Q[0, action] = reward + self.discount * np.amax(new_Q)

            if training_input is None:
                training_input = old_state
                target = old_Q
            else:
                training_input = np.vstack((training_input, old_state))
                target = np.vstack((target, old_Q))
        
        self.model.trainModel(training_input, target, batch=batch_size)
        if self.epsilon > 0:
            self.epsilon -= self.epsilon/self.iteration

    def seeQValues(self, state):
        """ print q value for the model """
        print("the state is")
        print(state)
        q_value = self.model.getQValue(state)
        print(q_value)