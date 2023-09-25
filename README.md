Deep Q-Learning Agent for CartPole Game in OpenAI's Gym

This repository contains an implementation of a Deep Q-Learning (DQN) agent to solve the CartPole-v1 environment provided by OpenAI's gym.

Key Features:

Deep Q-Network: The agent utilizes a neural network implemented using Keras. The model comprises two hidden layers each with 24 neurons and uses ReLU activation. The output layer has a linear activation function.
Exploration-Exploitation Strategy: The agent adopts an ε-greedy policy to strike a balance between exploration and exploitation. Over time, the exploration rate (ε) decays to shift the agent's approach from exploration to exploitation.
Experience Replay: Instead of learning online after every step, the agent stores experiences and samples from this memory for training. This approach ensures diverse training samples and breaks the correlation between consecutive experiences.
Training & Monitoring: The training loop simulates the CartPole game for a set number of episodes. After training, the agent's performance is recorded as a video and can be played back in a Jupyter notebook.
Dependencies:

numpy
gym
keras
IPython
Usage:
Simply run the provided code in a Python environment. If executed in a Jupyter notebook, the final section will display a video of the trained agent's performance in the CartPole-v1 environment.
