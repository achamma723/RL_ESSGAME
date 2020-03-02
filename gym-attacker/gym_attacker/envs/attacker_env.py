import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class Attacker(gym.Env):

    def __init__(self):
        self.state = None
        self.done = False
        self.reward = 0

    
    def init_params(self, K, initial_potential):
        self.K = K
        self.initial_potential = initial_potential
        self.weights = np.power(2.0, [-(self.K - i) for i in range(self.K + 1)])
        self.action_space = spaces.Discrete(self.K + 1)
        self.observation_space= spaces.MultiDiscrete([10]* ((K + 1)))
        
    def random_start(self):
        self.state = np.zeros(self.K + 1)
        potential = 0
        stop = False
        while (potential < self.initial_potential and not stop):
            possible = self.initial_potential - potential
            upper = self.K - 1 #upper is K-1 because K represents the top of the matrix which means end of the game
            while (2**(-(self.K-upper)) > possible):
                upper -=1
            if(upper < 0):
                stop = True
            else:
                if(upper > 0):
                    self.state[np.random.randint(0,upper)]+=1
                    potential = self.potential_fn(self.state)
                else:
                    self.state[0]+=1
                    potential = self.potential_fn(self.state)
        self.potential = potential
        self.state = np.array(self.state).astype(int)
        return self.state
    
    def potential_fn(self, A):
        return np.sum(A * self.weights)

    def reset(self):
        self.state = self.random_start()
        self.done = False
        self.reward = 0
        return self.state
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def split_level(self, idx):
        # given a level idx, checks to see what the splitting at that level
        # should be to ensure a more equal division
        weighted = self.state * self.weights
        l = self.state[idx] * self.weights[idx]
        pot1 = np.sum(weighted[:idx])
        pot2 = np.sum(weighted[idx+1:])

        if pot1 + l <= pot2:
            return (idx, self.state[idx], 0)

        # Case 1: where first state has more potential
        if pot1 + l > pot2:
            num_pieces = self.state[idx]
            if num_pieces == 0:
                return (idx, 0, 0)
            diff_pieces = (pot1 + l - pot2)/self.weights[idx]
            # divide by 2 as piece subtracted from A and added to B
            num_shift = min(int(diff_pieces/2), num_pieces)
            return (idx, num_pieces - num_shift, num_shift)     
            
    
    def propose_sets(self, action):
        """
        Function returns A, B -- proposed split of environment
        """
        # Create sets from action
        A = np.zeros(self.K+1).astype("int")
        B = np.zeros(self.K+1).astype("int")
        idx = action

        sidx, sA, sB = self.split_level(idx)

        A[:idx+1] = self.state[:idx+1]
        B[idx+1:] = self.state[idx+1:]
        A[sidx] = sA
        B[sidx] = sB

        rand = np.random.binomial(1, 0.5)
        if rand:
            return A, B
        else:
            return B, A
        
    def change_state(self, new_state):
        self.state = new_state
        
    def erase(self, A):
        """Function to remove the partition A from the game state

        Arguments:
            A {list} -- The list representing the partition we want to remove
        """

        self.state = [z - a for z, a in zip(self.state, A)]
        self.state = np.array([0] + self.state[:-1]).astype(int) 
      
    
    def defense_play(self, A, B):
        potA = self.potential_fn(A)
        potB = self.potential_fn(B)
        if (potA >= potB):
            self.erase(A)
            return 'A'
        else:
            self.erase(B)
            return 'B'


    def check(self):
        """Function to chek if the game is over or not.

        Returns:
            int -- If the game is not over returns 0, otherwise returns -1 if the defender won or 1 if the attacker won.
        """

        if (sum(self.state) == 0):
            return -1
        elif (self.state[-1] >=1 ):
            return 1
        else:
            return 0
    
    def step(self, target):
        A, B = self.propose_sets(target)
        erased = self.defense_play(A, B)
        win = self.check()
        if(win):
            self.done = True
            self.reward = win

        return self.state, self.reward, self.done, {'A':A, 'B':B, 'Erased':erased}


    def render(self):
        for j in range(self.K + 1):
            print(self.state[j], end = " ")
        print("")
