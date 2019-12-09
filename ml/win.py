
import numpy as np

def won(state, indx, value):
    smatch = 3
    srow = 3
    scol = 3
    """ check the status of game """
    state[indx] = value
    state = state.reshape(srow, scol)
    tick = np.array([indx // scol, indx % scol])
    start = np.copy(tick)
    print(state, tick, start)
    print("#######")
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
        for k in range(smatch):
            match = 0
            tick[0] = tick[0] - row * k
            tick[1] = tick[1] - col * k
            print(tick, i, k)
            if min(tick) < 0 or tick[1] >= scol:
                tick = np.copy(start)
                break
            #value = state[tick[0], tick[1]]
            for j in range(smatch):
                x = tick[0] + row * j
                y = tick[1] + col * j
                print(x, y, j)
                if x >= srow or y >= scol or y < 0:
                    tick = np.copy(start)
                    break
                if value == state[x, y]:
                    match += 1
                else:
                    tick = np.copy(start)
                    break
                print(state[x, y])
            if match == smatch:
                done = 1
                return done
            tick = np.copy(start)
    return done

def main():
    state = np.array([0,0,1, 1,0,2, 0,2,0])
    if won(state, 0, 2):
        print("win")

if __name__ == '__main__':
    main()