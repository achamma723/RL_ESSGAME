import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Defender(gym.Env):


    def __init__(self):
        self.state = None
        self.game_state = None
        self.done = False
        self.reward = 0
        self.action_space = spaces.Discrete(2)
        
    def init_params(self, K, initial_potential, adverse_set_prob, disj_supp_prob, model_state):
        self.K = K
        self.initial_potential = initial_potential
        self.weights = np.power(2.0, [-(self.K - i) for i in range(self.K + 1)])
        self.observation_space= spaces.MultiDiscrete([10]* (2*K+2))
        self.adverse_set_prob = adverse_set_prob
        self.disj_supp_prob = disj_supp_prob
        self.model_state = model_state
        
    def random_start(self):
        self.game_state = np.zeros(self.K + 1)
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
                    self.game_state[np.random.randint(0,upper)]+=1
                    potential = self.potential_fn(self.game_state)
                else:
                    self.game_state[0]+=1
                    potential = self.potential_fn(self.game_state)
        self.potential = potential
        self.game_state = np.array(self.game_state).astype(int)
        return self.game_state

    def potential_fn(self, A):
        return np.sum(A * self.weights)

    def reset(self):
        self.game_state = self.random_start()
        self.done = False
        self.reward = 0
        A, B = self.propose_sets()
        self.state = np.concatenate([A,B])
        return self.state
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def propose_sets(self):
        # picks optimal method of proposing sets
        # with probability self.difficulty, otherwise
        # picks sets at random

        # if only few pieces left, play optimally
        num_idxs = np.sum(self.game_state > 0)
        if num_idxs <= 3:
            A, B = self.propose_sets_opt()

        else:
            propose_types = ["adversarial", "disj_support", "opt"]
            idx_arr = np.random.multinomial(1, [self.adverse_set_prob, self.disj_supp_prob, 1 - self.adverse_set_prob - self.disj_supp_prob])
            idx = np.argmax(idx_arr)
            propose_type = propose_types[idx]

            if propose_type == "adversarial":
                if(len(self.model_state) == 0):
                    assert "cannot propose adversarial sets with empty model state"
                elif(len(self.model_state) < self.K + 1):
                    assert "model state doesn't have the required length"
                else:
                    self.model_state = np.array(self.model_state)
                A, B = self.propose_sets_adversarial()
            elif propose_type == "disj_support":
                A, B = self.propose_sets_disj_support()
            elif propose_type == "opt":
                A, B = self.propose_sets_opt()
            else:
                 raise ValueError("unsupported set propose_type")

        return A, B
    
    def propose_sets_disj_support(self):
        # proposes sets of disjoint support
        # with varying potential split
        
        A = np.zeros(self.K+1, dtype=int)
        B = np.zeros(self.K+1, dtype=int)

        nonzeros = np.where(self.game_state > 0)[0]
        thresholds = [1./3, 5./16, 14./32]
        _ = np.random.multinomial(3, [0.8, 0.1, 0.1])
        _ = np.argmax(_)
        threshold = thresholds[_]
        idxs = nonzeros[np.random.permutation(len(nonzeros))]
        
        potA = self.potential_fn(A)
        potB = self.potential_fn(B)
        for idx in idxs:
            l_pieces = self.game_state[idx]
            # check to see what potential of pieces is
            # if potential very large, fraction, equally divide
            if l_pieces*self.weights[idx] >= self.potential/2.:
                # try to equally divide
                if l_pieces % 2 == 0:
                    pieces = int(l_pieces/2)
                    A[idx] += pieces
                    B[idx] += pieces
                    potA += (l_pieces*self.weights[idx])/2.
                    potB += (l_pieces*self.weights[idx])/2.
                else:
                    A[idx] += int(l_pieces/2)
                    B[idx] += (int(l_pieces/2) + 1)
                    potA += int(l_pieces/2)*self.weights[idx]
                    potB += (int(l_pieces/2) + 1)*self.weights[idx]

            else:
                if potA >= threshold*self.potential:
                    B[idx] += l_pieces
                    potB += l_pieces*self.weights[idx]
                else:
                    A[idx] += l_pieces
                    potA += l_pieces*self.weights[idx]
        # vary which of A or B is underweighted set
        p = np.random.uniform(low=0, high=1)
        if p >= 0.5:
            return B, A
        else:
            return A, B
        
    def propose_sets_adversarial(self):
        # proposes adversarial sets for current
        # weight of model

        A = np.zeros(self.K+1, dtype=int)
        B = np.zeros(self.K+1, dtype=int)
        diff = (self.game_state*(self.weights - (self.model_state/np.abs(self.model_state[-2]))))[:-1]

        # want to fill up set with most underweighted terms first
        idxs = np.argsort(diff)[::-1]
        idxs = np.append(idxs, self.K)

        # fill up set to get around half of existing potential
        threshold = self.potential_fn(self.game_state)/2

        for i in idxs:
            
            # check potential to break
            potA = self.potential_fn(A)
            if potA > threshold + max(self.weights[0], 1e-8):
                break
            
            # get the number of pieces
            l_pieces = self.game_state[i]
            if l_pieces == 0:
                continue
            # add to A   
            num_pieces = np.ceil((threshold + max(self.weights[0], 1e-8) - potA)/self.weights[i])
            A[i] += np.min([l_pieces, num_pieces])

        # B is the complement of A
        B = self.game_state - A
        assert (np.all(B >= 0)), print("state, A and B", self.game_state, A, B)

        # vary which of A or B is underweighted set
        p = np.random.uniform(low=0, high=1)
        if p >= 0.5:
            return B, A
        else:
            return A, B

    def equal_divide(self, A, B, potA, potB, l, l_weight, weight, l_pieces):
        # divides up pieces when potA, potB are equal except off by 1

        if l_pieces == 0:
            return A, B

        if l_pieces % 2 == 0:
            A[l] += l_pieces/2
            B[l] += l_pieces/2

        else:
            larger = np.ceil(l_pieces/2)
            smaller = np.floor(l_pieces/2)
            assert larger + smaller == l_pieces, print("division incorrect", larger, smaller, l_pieces)

            if potA < potB:
                A[l] += larger
                B[l] += smaller

            elif potB < potA:
                A[l] += smaller
                B[l] += larger

            else:
                prob_A = np.random.binomial(1, 0.5)
                if prob_A:
                    A[l] += larger
                    B[l] += smaller
                else:
                    A[l] += smaller
                    B[l] += larger

        return A, B

    def propose_sets_opt(self):
        # proposes optimial choices of sets
        # by givng two sets with potential both
        # >= 1/2 (can do by splitting lemma)

        A = np.zeros(self.K+1, dtype=int)
        B = np.zeros(self.K+1, dtype=int)

        levels = [i for i in range(self.game_state.shape[0])]
        levels.reverse()

        for l in levels:
            l_pieces = self.game_state[l]
            if l_pieces == 0:
                continue

            weight = self.weights[l]
            l_weight = l_pieces*weight
            potA = self.potential_fn(A)
            potB = self.potential_fn(B)
            
            # divide equally at that level if potentials are equal
            if potA == potB:
                A, B = self.equal_divide(A, B, potA, potB, l, l_weight, weight, l_pieces)
            
            # if potentials are not equal
            else:
                diff = np.abs(potA - potB)
                num_pieces = np.ceil(diff/weight).astype("int")

                # if the number of pieces which are the difference is less than l_pieces
                if num_pieces <= l_pieces:
                    diff_pieces = num_pieces

                    if potA < potB:
                        A[l] += diff_pieces
                    else:
                        B[l] += diff_pieces

                    l_pieces -= diff_pieces
                    A, B = self.equal_divide(A, B, potA, potB, l, l_weight, weight, l_pieces)

                else:
                    if potA < potB:
                        A[l] += l_pieces
                    else:
                        B[l] += l_pieces

        return A, B
        
    def split(self, A):
        B = [z - a for z, a in zip(self.game_state, A)]
        return A, B

    def erase(self, A):
        """Function to remove the partition A from the game state

        Arguments:
            A {list} -- The list representing the partition we want to remove
        """

        self.game_state = [z - a for z, a in zip(self.game_state, A)]
        self.game_state = np.array([0] + self.game_state[:-1]).astype(int)
        
        
    def change_state(self, new_state):
        self.state = new_state
        tmp1 = new_state[:self.K + 1]
        tmp2 = new_state[self.K + 1:]
        self.game_state = tmp1 + tmp2
        
    def check(self):
        """Function to chek if the game is over or not.

        Returns:
            int -- If the game is not over returns 0, otherwise returns 1 if the defender won or -1 if the attacker won.
        """

        if (sum(self.game_state) == 0):
            return 1
        elif (self.game_state[-1] >=1 ):
            return -1
        else:
            return 0

    def step(self, target):
        A = self.state[: self.K + 1]
        B = self.state[self.K + 1 :]
        if (target == 0):
            self.erase(A)
        else:
            self.erase(B)
        win = self.check()
        if(win):
            self.done = True
            self.reward = win
            A, B = self.propose_sets()
            self.state = np.concatenate([A,B])     

        else:
            A, B = self.propose_sets()
            self.state = np.concatenate([A,B])
            
        return self.state, self.reward, self.done, {}

    def render(self):
        for j in range(self.K + 1):
            print(self.game_state[j], end = " ")
        print("")
