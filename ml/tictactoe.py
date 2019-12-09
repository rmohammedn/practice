import numpy as np
from deepqmodel import DeepQModel
from qagent import QAgent

class TicTacToe:
    """ the 3 X 3 tic-tac-toe game """

    def __init__(self, row=3, col=3, match=3):
        """ initialize tictactoe board """
        self.state = np.zeros(row * col)
        self.row = row
        self.col = col
        self.match = match
        self.val1 = 1
        self.val2 = 2
        self.winner = None
        self.drow = None
    
    def nextState(self, action, val):
        """ tick (action) is the location where players add new entry """
        indx = action
        #print(action, self.state)
        if self.state[indx] != 0:
            reward = -10
            return reward
        elif val == self.val1:
            self.state[indx] = val
            if self.won(indx, val):
                reward = 5
                self.winner = val
                return reward
            elif self.checkDrow():
                reward = 1
                return reward
            else:
                reward = 0
                return reward
        elif val == self.val2:
            self.state[indx] = val
            if self.won(indx, val):
                reward = 5
                self.winner = val
                return reward
            elif self.checkDrow():
                reward = 1
                return reward
            else:
                reward = 0
                return reward
        else:
            print("not valid input")

    def checkDrow(self):
        """ check the match is drow """
        indx = np.where(self.state == 0)
        if len(indx[0]) == 0:
            self.drow = 1
            return 1
        else:
            return 0
    
    def won(self, indx, value):
        """ check the status of game """
        state = self.state.reshape(self.row, self.col)
        tick = np.array([indx // self.col, indx % self.col])
        start = np.copy(tick)
        #print(state, indx)
        done = 0
        for i in range(4):
            if i == 0:
                row = 1
                col = 0
            elif i == 1:
                row = 0
                col = 1
            elif i == 2:
                row = 1
                col = -1
            else:
                row = 1
                col = 1
            for k in range(self.match):
                match = 0
                tick[0] = tick[0] - row * k
                tick[1] = tick[1] - col * k
                if min(tick) < 0 or tick[1] >= self.col:
                    tick = np.copy(start)
                    break
                #value = state[tick[0], tick[1]]
                for j in range(self.match):
                    x = tick[0] + row * j
                    y = tick[1] + col * j
                    if x >= self.row or y >= self.col or y < 0:
                        tick = np.copy(start)
                        break
                    if value == state[x, y]:
                        match += 1
                    else:
                        tick = np.copy(start)
                        break
                if match == self.match:
                    done = 1
                    return done
                tick = np.copy(start)
        return done

    def reset(self):
        self.state = np.zeros(self.row * self.col)
        self.winner = None
        self.drow = None


def main():
    """ combine all the module """

    arena = TicTacToe()
    model = DeepQModel()
    #model.loadModel("tictactoe1.h5")
    player1 = QAgent(9, model)
    player2 = QAgent(9, model)
    arena.reset()
    drow = 0
    win1 = 0
    win2 = 0
    iteration = 500000
    batch_size = 100
    for i in range(iteration):
        #print(i)
        val = i % 2
        if val == 0:
            val = 2
        state = arena.state
        old_state = state.reshape(1, len(state))
        if val == 1:
            action = player1.getNextAction(old_state)
            reward = arena.nextState(action, val)
            new_state = arena.state.reshape(1, len(state))
            player1.remember(old_state, action, new_state, reward)
            #print(player1.epsilon)
            if len(player1.memory) > batch_size:
                player1.train(batch_size)
            if arena.winner == val:
                #print(new_state.reshape(3,3), "win1", action)
                reward = -5
                player2.remember(old_state, action, new_state, reward)
                arena.reset()
                win1 += 1
            if arena.drow == 1:
                #print(new_state.reshape(3,3))
                player2.remember(old_state, action, new_state, reward)
                arena.reset()
                drow += 1
        else:
            action = player2.getNextAction(old_state)
            reward = arena.nextState(action, val)
            new_state = arena.state.reshape(1, len(state))
            player2.remember(old_state, action, new_state, reward)
            #print(player2.epsilon)
            if len(player2.memory) > batch_size:
                player2.train(batch_size)
            if arena.winner == val:
                #print(new_state.reshape(3,3), "win2", action)
                reward = -5
                player1.remember(old_state, action, new_state, reward)
                arena.reset()
                win2 += 1
            if arena.drow == 1:
                #print(new_state.reshape(3,3))
                player1.remember(old_state, action, new_state, reward)
                arena.reset()
                drow += 1
        if i % 500 == 0:
            player1.seeQValues(old_state)
        if i % 1000 == 0:
            print("player1 win", win1)
            print("player2 win", win2)
            print("drow", drow)
            print(player1.epsilon, player2.epsilon)
            print("west = ", 1000 - (win1 + win2 + drow)*9)
            win1 = win2 = drow = 0
            print("==============================")
    #player2.saveQModel("tictactoe2.h5")
    #player1.saveQModel("tictactoe1.h5")

if __name__ == '__main__':
    main()