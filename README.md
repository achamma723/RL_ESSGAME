# dqn-attack-defense

Implementation of Deep Q Learning to Solve Erdos-Selfridge-Spencer Games


## Article

Based on the article:

[Can Deep Reinforcement Learning Solve Erdos-Selfridge-Spencer Games?
Maithra Raghu, Alex Irpan, Jacob Andreas, Robert Kleinberg, Quoc V. Le, Jon Kleinberg
](https://arxiv.org/pdf/1711.02301.pdf)

## How to use

  1. Install all packages in requirements.txt using pip install -r requirements.txt
  2. Install the gym environments using 
  ```
  pip install -e gym-defender/
  pip install -e gym-attacker/
  pip install -e stable-baslines/
  ```
  3. You can choose all the possible values of K (the level) and the potential function.
  You will find the main usages of our environnements and agents in [a notebook (Notebook.ipynb)](Notebook.ipynb).

## Baselines

OpenAI Baselines : https://github.com/openai/baselines

Stable Baselines Guide : https://pythonawesome.com/a-fork-of-openai-baselines-implementations-of-reinforcement-learning-algorithms/
  
## Authors

- Ahmad CHAMMA
- Hadi ABDINE
