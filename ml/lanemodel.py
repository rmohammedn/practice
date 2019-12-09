import numpy as np
from deepqmodel import DeepQModel
from qagent import QAgent


class Location:
    # .............#
    def __init__(self, x=0, y=0, angle=np.pi/2):
        self.x = x
        self.y = y
        self.angel = angle

class Vehicle:
    # create vehicles object with its specifications
    def __init__(self, x, y, a=None, length=4, width=2, lane=0):
        self.length = length
        self.width  = width
        self.lane = lane
        self.location = Location(x, y, a)


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
        self.action_size = 12
        self.car_list = self.getVehicles(3)
        #self.state = np.array([self.a0, self.a1, self.a2, self.d])
    
    def getVehicles(self, num):
        """ spawn num number of vehicles """
        return np.array([Vehicle(0, 5), Vehicle(5, 5), Vehicle(10, 5)])
  
    def getState(self):
        """ create simple state """
        self.state_size = 4
        return np.array([self.a0, self.a1, self.a2, self.d])

    def getDistance(self, car, line):
        """ perpendicular distance between car and line """
        a0 = self.left_end + line * self.d
        x = (car.location.x + 2 * self.a2 * (self.a1 - car.location.y)) / (1 + 4 * self.a1**2)
        y = self.a2 * x * x + self.a1 * x + a0
        dist = (car.location.x - x)**2 + (car.location.y - y)**2
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
            else:
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
            if self.no_lane > 1:
                self.no_lane -= 1
            reward = self.getReward()
            return reward
        else:
            if self.no_lane > 1:
                self.no_lane -= 1
                self.left_end += self.d
            reward = self.getReward()
            return reward


def main():
    arena = LaneModel()
    action_size = arena.action_size
    arena.getState()
    state_size = arena.state_size
    model = DeepQModel(state_size, action_size)
    agent = QAgent(action_size, model)
    iteration = 10000
    batch_size = 50
    for i in range(iteration):
        state = arena.getState()
        old_state = state.reshape(1, state_size)
        action = agent.getNextAction(old_state)
        reward = arena.nextState(action)
        new_state = arena.getState().reshape(1, state_size)
        agent.remember(old_state, action, new_state, reward)
        if len(agent.memory) > batch_size:
            agent.train(batch_size)
        if i % 100 == 0:
            agent.seeQValues(old_state)
    print(arena.state)

if __name__=="__main__":
    main()