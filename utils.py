import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import load_results, ts2xy
import time
from IPython.display import clear_output

def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def extract_results(log_folder, title='Learning Curve'):
    """
    extract the results from the monitor file saved in tmp

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = movingAverage(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]
    
    return x, y

def potential_fn(K, A):
    return np.sum(A * np.power(2.0, [-(K - i) for i in range(K + 1)]))
    
def random_start(K, initial_potential):
    state = np.zeros(K + 1)
    potential = 0
    stop = False
    while (potential < initial_potential and not stop):
        possible = initial_potential - potential
        upper = K - 1 #upper is K-1 because K represents the top of the matrix which means end of the game
        while (2**(-(K - upper)) > possible):
            upper -=1
        if(upper < 0):
            stop = True
        else:
            if(upper > 0):
                state[np.random.randint(0,upper)]+=1
                potential = potential_fn(K, state)
            else:
                state[0]+=1
                potential = potential_fn(K, state)
    state = np.array(state).astype(int)
    return state

def printViz(viz, k, N):
    plt.figure(figsize=(12,8))
    for el in range(k + 1):
        plt.axhline(y=el - 0.5, linestyle='-')
    for el in range(N + 1):
        plt.axvline(x=el - 0.5, linestyle='-')
        
    plt.xticks(np.arange(N+1))
    plt.yticks(np.arange(k+1))
    plt.imshow(viz, origin='lower', cmap='gray', interpolation="none")
    plt.show()
        
def simulate_trainedAttackerDefender(attack, defense, initial_potential):
    K = attack.K
    initial_state = random_start(K, initial_potential)
    N = np.sum(initial_state)
    viz = np.zeros((K + 1 ,N))
    for ind, el in enumerate(initial_state.reshape(-1, 1)):
        viz[ind, :int(el)] = np.ones(int(el))
    print("Start..")
    print("Initial state:", initial_state)
    printViz(viz, K, N)
    time.sleep(2)

    done = False
    state = initial_state
    while True: 
        clear_output(wait=True)
        print("Attacker turn..")
        state = np.reshape(np.array(state), [1, attack.state_size])

        attack.envs.env_method('change_state', state[0])
        action, _states = attack.model.predict(state)
        _, reward, done, _a = attack.envs.step(action)
                
        partitionA = _a[0]['A']
        partitionB = _a[0]['B']
        print("Partitions : ", partitionA, partitionB)
        
        viz = np.zeros((K + 1 ,N))
        for i in range(K):
            ind1 = int(partitionA[i])
            ind2 = int(partitionB[i])
            viz[i,ind1:(ind1 + ind2)] = np.ones(ind2) * 0.3
            viz[i,:ind1] = np.ones(ind1) * 0.2
        printViz(viz, K, N)
        time.sleep(2)
        
        viz = np.zeros((K + 1 ,N))
        clear_output(wait=True)
        
        print("Defender turn..")
        state = np.concatenate([partitionA, partitionB])
        defense.envs.env_method('change_state', state)
        
        state = np.reshape(np.array(state), [1, defense.state_size])
        
        action, _states = defense.model.predict(state)
        state, reward, done, _d = defense.envs.step(action)
        
        if ('terminal_observation' in _d[0].keys()):
            state = _d[0]['terminal_observation']
            state = np.reshape(np.array(state), [1, defense.state_size])
        
        A = state[0][: K + 1]
        B = state[0][K + 1 :]
        
        if(action[0] == 1):
            print("Defender keeps:", partitionA)
            #if partitionA[-1] > 0 or np.sum(partitionA) == 0:
            #    done = True
        else:
            print("Defender keeps:", partitionB)
            #if partitionB[-1] > 0 or np.sum(partitionB) == 0:
            #    done = True
        for ind, el in enumerate((A+B).reshape(-1, 1)):
            if ind > 0:
                viz[ind, :int(el)] = np.ones(int(el))
    
        printViz(viz, K, N)
        time.sleep(2)
        
        state = A + B
        if np.sum(state) == 0:
            print("Defender wins!!")
            break
        elif state[-1] >= 1:
            print("Attacker wins!!")
            break