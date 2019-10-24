import tensorflow as tf
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
            reward = self.max_r
        else:
            self.state = 0
            reward = self.min_r
        return self.state, reward

    def reset(self):
        """ reset the state """
        self.state = 0
        return self.state

class DeepQAgent:
    """ ...... """

    def __init__(self, learning_rate=0.1, discount=0.9, expo_rate=1.0, hidden=4, iteration=50):
        """Initializing all the parameter required for this deep Q model the tensorflow graph is also created"""

        self.learning_rate = learning_rate
        self.discount = discount
        self.expo_rate = expo_rate
        self.iteration = iteration
        self.input_size = 5
        self.output_size = 2
        self.sess = tf.Session()
        self.make_model(hidden)
        self.sess.run(self.initializer)

    def make_model(self, hl):
        """ creating tensorflow graph """

        self.input_layer = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        hdl1 = tf.layers.dense(self.input_layer, hl, activation=tf.sigmoid, kernel_initializer=tf.constant_initializer(np.zeros((self.input_size, hl))))
        hdl2 = tf.layers.dense(hdl1, hl, activation=tf.sigmoid, kernel_initializer=tf.constant_initializer(np.zeros((hl, self.output_size))))
        self.output_layer = tf.layers.dense(hdl2, self.output_size)

        self.target_output = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        loss = tf.losses.mean_squared_error(self.target_output, self.output_layer)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(loss)

        self.initializer = tf.global_variables_initializer()

    def getQValue(self, state):
        """ return Q value whenever required """

        return self.sess.run(self.output_layer, feed_dict={self.input_layer: self.toOneHot(state)})

    def toOneHot(self, state):
        """ return one hote of any input like 2 is 0 0 1 0 0 """

        one_hote = np.zeros((1, 5))
        one_hote[0, state] = 1
        return one_hote

    def getNextAction(self, state):
        """ it will return next state depending on policy """

        if np.random.random() > self.expo_rate:
            return self.greedyPolicy(state)
        else:
            return self.randomAction()

    def greedyPolicy(self, state):
        """ it will take the next state with highest Q value """

        return np.argmax(self.getQValue(state))

    def randomAction(self):
        """ ... """

        if np.random.random() > 0.5:
            return 1
        else:
            return 0

    def train(self, old_state, action, new_state, reward):
        """ will train the model """

        old_Q = self.getQValue(old_state)
        new_Q = self.getQValue(new_state)
        #print(old_Q, new_Q)

        old_Q[0, action] = reward + self.discount * np.amax(new_Q)

        training_input = self.toOneHot(old_state)
        target = old_Q

        self.sess.run(self.optimizer, feed_dict={self.input_layer: training_input, self.target_output: target})

        if self.expo_rate > 0:
            self.expo_rate -= self.expo_rate/2000

def main():
    """ ... """

    arena = Simulator()
    agent = DeepQAgent()
    arena.reset()

    for i in range(agent.iteration):
        state = arena.state
        action = agent.getNextAction(state)
        new_state, reward = arena.nextState(action)
             
        for i in range(5):
            Q_value = agent.getQValue(state)
            print(Q_value)
        #print(state, action, new_state, reward)
        

if __name__== "__main__":
    main()
