import os

import gym
import gym_attacker

from stable_baselines.common.policies import FeedForwardPolicy as FFP_common
from stable_baselines.deepq.policies import FeedForwardPolicy as FFP_DQ

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor

from stable_baselines import PPO2, DQN, A2C

from stable_baselines.results_plotter import load_results, ts2xy

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time
from IPython.display import clear_output

class MLP_PPO(FFP_common):
    def __init__(self, *args, **kwargs):
        super(MLP_PPO, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[300, 300], vf=[300, 300])],
                                           feature_extraction="mlp")
        
class MLP_DQN(FFP_DQ):
    def __init__(self, *args, **kwargs):
        super(MLP_DQN, self).__init__(*args, **kwargs,
                                           layers=[300, 300],
                                           layer_norm=False,
                                           feature_extraction="mlp")

class MLP_A2C(FFP_common):
    def __init__(self, *args, **kwargs):
        super(MLP_A2C, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[300, 300], vf=[300, 300])],
                                           feature_extraction="mlp")
        
class Attack:
    def __init__(self, method, K=5, P=0.95):
        self.method = method
        self.K = K
        self.state_size        = self.K + 1
        self.action_size       = self.K + 1
        self.reward            = []
        
        env_name = 'ErdosAttack-v0'
        
        self.log_dir = "/tmp/gym_attack/"
        os.makedirs(self.log_dir, exist_ok=True)
        
        env = gym.make(env_name)
        env.init_params(K, P)
        env = Monitor(env, self.log_dir, allow_early_resets=True)
        self.envs = DummyVecEnv([lambda: env])
        
        if method=='PPO':
            self.model = PPO2(MLP_PPO, self.envs, verbose=0)
        elif method=='DQN':
            self.model = DQN(MLP_DQN, self.envs, verbose=0)
        elif method=='A2C':
            self.model = A2C(MLP_A2C, self.envs, verbose=0)
        else:
            raise Exception("Erreur ! Méthode: 'PPO' ou 'DQN' ou 'A2C")
        print("Model Initialized !")
        
        self.best_mean_reward, self.n_steps = -np.inf, 0
        
    
    def callback(self, _locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
       
        # Print stats every 1000 calls
        if (self.n_steps + 1) % 1000 == 0:
            # Evaluate policy performance
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                print(x[-1], 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                
        self.n_steps += 1
        return True
    
    def learn(self, timesteps=10000):
        self.model.learn(total_timesteps = timesteps, callback = self.callback)
        print("======\n{} LEARNING DONE ATTACK\n======".format(self.method))
            
    def printViz(self, viz, k, N):
        plt.figure(figsize=(12,8))
        for el in range(k + 1):
            plt.axhline(y=el - 0.5, linestyle='-')
        for el in range(N + 1):
            plt.axvline(x=el - 0.5, linestyle='-')
            
        plt.xticks(np.arange(N+1))
        plt.yticks(np.arange(k+1))
        plt.imshow(viz, origin='lower', cmap='gray', interpolation="none")
        plt.show()
    
    def simulate_trainedAttacker(self):
        initial_state = self.envs.reset()
        N = np.sum(initial_state)
        viz = np.zeros((self.K + 1 , N))

        for ind, el in enumerate(initial_state[0].reshape(-1, 1)):
            viz[ind, :int(el)] = np.ones(int(el))
            
        print("Start..")
        print("Initial state:", initial_state[0])
        self.printViz(viz, self.K, N)
        time.sleep(2)
        state = np.reshape(np.array(initial_state), [1, self.state_size])
        done = False
        
        while not done:
            clear_output(wait=True)
            print("Attacker turn..")
            action, _states = self.model.predict(state)
            state, reward, done, _ = self.envs.step(action)

            partitionA = _[0]['A']
            partitionB = _[0]['B']
            print("Partitions : ", partitionA, partitionB)
            
            viz = np.zeros((self.K + 1 ,N))
            for i in range(self.K):
                ind1 = int(partitionA[i])
                ind2 = int(partitionB[i])
                viz[i,ind1:(ind1 + ind2)] = np.ones(ind2) * 0.3
                viz[i,:ind1] = np.ones(ind1) * 0.2
            self.printViz(viz, self.K, N)
            time.sleep(2)
            
            viz = np.zeros((self.K + 1 ,N))
            clear_output(wait=True)
            print("Defender turn..")
            
            if ('terminal_observation' in _[0].keys()):
                state = _[0]['terminal_observation']
                state = np.reshape(np.array(state), [1, self.state_size])
                
            if(_[0]['Erased'] == 'A'):
                print("Defender keeps:", partitionB)
            else:
                print("Defender keeps:", partitionA)
            for ind,el in enumerate(state.reshape(-1, 1)):
                if ind > 0:
                    viz[ind, :int(el)] = np.ones(int(el))

            self.printViz(viz, self.K, N)
            time.sleep(2)
            
            if done:
                if reward == -1:
                    print("Defender wins!!")
                else:
                    print("Attacker wins!!")
                
    def run(self, nb_episodes = 1000):
        self.reward = []
        self.nb_episodes = nb_episodes
        
        for index_episode in range(nb_episodes):
            state = self.envs.reset()
            state = np.array(state)
            done = False
            steps = 0
            while not done:
                 action, _states = self.model.predict(state)
                 next_state, reward, done, _ = self.envs.step(action)
                 next_state = np.array(next_state)
                 state = next_state
                 steps += 1
            if index_episode %100 == 0:
                print("Episode {}#; \t Nb of steps: {}; \t Reward: {}.".format(index_episode, steps + 1, reward))
            if index_episode > 0:
                self.reward += [((self.reward[-1] * len(self.reward)) + reward) / (len(self.reward) + 1)]
            else:
                self.reward += [reward]
