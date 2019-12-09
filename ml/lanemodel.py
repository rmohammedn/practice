import numpy as np
from deepqmodel import DeepQModel
from qagent import QAgent

"""
class Location:
    self.x = 0
    self.y = 0
    self.angel = np.pi

class Vehicle:
    # create vehicles object with its specifications
    def __init__(self, length=4, width=2, lane=0, location=Location()):
        self.length = length
        self.width  = width
        self.lane = lane
        self.location = location
"""

class LaneModel:
    """ lane paramerters """
    def __init__(self):
        self.a0 = 0
        self.a1 = 0
        self.a2 = 0
        self.d  = 3
        self.change = 0.2
        self.left_end = 0
        self.no_lane = 1
        self.car_list = [(0, 5, np.pi/2), (5, 5, np.py/2), (10, 5, np.py/2)]
        self.state_size = 4
        self.action_size = 12
        self.state = np.array([self.a0, self.a1, self.a2, self.d])

    def getDistance(self, car, line):
        """ perpendicular distance between car and line """
        a0 = self.left_end + line * self.d
        x = (car[0] + 2 * self.a2 * (self.a1 - car[1])) / (1 + 4 * self.a1**2)
        y = self.a2 * x * x + self.a1 * x + a0
        dist = (car[0] - x)**2 + (car[1] - y)**2
        return dist

    def getReward(self):
        """ calculate reward """
        reward = 0
        dist_list = np.array([])
        distance = np.array([])
        for car in self.car_list:
            for lane in range(self.no_lane):
                dist = self.getDistance(car, lane)
                dist_list = np.append(dist_list, dist)
            min_dist = np.min(dist_list)
            distance = np.append(distance, min_dist)
        
        for dist in distance:
            if dist == 0:
                reward += 10
            elif dist <= 0.5 and dist > 0:
                reward += 0
            else:
                reward = reward - dist
        return reward

    def nextState(self, action):
        """ for given action find next state """
        """ total number of actions : [-0, 0, 1, -1, 2, -2, d, -d, nrl, nll, rrl, rll] """
        change = self.change
        if action == 0:
            self.a1 -= change
            reward = self.getReward()
            return reward
        elif action == 1:
            self.a1 += change
            reward = self.getReward()
            return reward
        elif action == 2:
            self.a2 -= change / 2
            reward = self.getReward()
            return reward
        elif action == 3:
            self.a2 += change / 2
            reward = self.getReward()
            return reward
        elif action == 4:
            self.a0 += change
            reward = self.getReward()
            return reward
        elif action == 5:
            self.a0 -= change
            reward = self.getReward()
            return reward
        elif action == 6:
            n = self.left_end / self.d
            self.d += change / 5
            self.left_end = n * self.d
            reward = self.getReward()
            return reward
        elif action == 7:
            n = self.left_end / self.d
            self.d -= change
            self.left_end = n * self.d
            reward = self.getReward()
            return reward
        elif action == 8:
            self.no_lane += 1
            reward = self.getReward()
            return reward
        elif action == 9:
            self.no_lane += 1
            self.left_end -= self.d
            reward = self.getReward()
            return reward
        elif action == 10:
            self.no_lane -= 1
            reward = self.getReward()
            return reward
        else:
            self.no_lane -= 1
            self.left_end += self.d
            reward = self.getReward()
            return reward


def main():
    arena = LaneModel()
    action_size = arena.action_size
    state_size = arena.state_size
    model = DeepQModel(state_size, action_size)
    agent = QAgent(action_size, model)
    iteration = 100
    batch_size = 10
    for i in range(iteration):
        state = arena.state
        old_state = state.reshape(1, state_size)
        action = agent.getNextAction(old_state)
        reward = arena.getReward(action)
        new_state = arena.state
        agent.remember(old_state, action, new_state, reward)
        if len(agent.memory) > batch_size:
            agent.train(batch_size)
