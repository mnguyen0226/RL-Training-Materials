"""
    Name: Minh T. Nguyen
    1/13/2021
    Algorithms Credit:
        Foundations of Deep RL
        Chapter 2

    About: Implementation of the basic Reinforce algorithms with CartPole

    Algorithms Explained:
    1/ Initialize learning rate
    2/ Initialize weight of a policy network pi
    3/
    For episodes 0....Max eps, do:
        Sample a trajectory t = s, a, t
        Set the policy gradient = 0
        For t = 0,....t do:
            Calculate Rt
            Update the policy gradient
        end
        Update weight
    End

    Note: make sure to create anaconda and install python package with "python setup.py develop" on terminal
    -----------------------------------------------------------------------------------
    * How to construct a policies with Pytorch?
    - The output of the neural network are transformed into an action probability distribution that is used to sample an action
    - Key idea: that probability distribution can be parameterized either by:
        + enumberating the full probabilioties for discrete distribution
        + specifying the mean and standard deviation of continuous distribution such as Normal distribution
    - These prob distribition param can be learned and output by the neural network
    - To output the action:
        + Use the policy network to compute the prob distribution param from a state
        + Use these param to construct an action prob distribution
        + Use the action prob distribution to sample an action and computer the action log prob
"""
from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize learning rate
gamma = 0.99

# Class for Monte Carlo Randomization - This is a policy network.
# Pi constructs the policy network that is a simple one layer MLP with 64 hidden units
class Pi(nn.Module):
    def __init__(self, in_dim, out_dim): # initialize network, model, reset state, and training mode
        super(Pi, self).__init__()
        layers = [ # 2 Dense layers with relu activation
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers) # variable that create a sequential network, this should return the output of the network
        self.onpolicy_reset()
        self.train()  # set training mode, global function

    def onpolicy_reset(self): # Clear up log prob and rewards since data is unreuseable
        self.log_probs = []
        self.rewards = []

    # Calculate the forward pass and return the output value of the output
    def forward(self, x): # Forward pass
        pdparam = self.model(x) #
        return pdparam

    # Method to produce action taking in the current state of the object and return the actions
    """
    Construct a discrete policy:
        1/ Given a policy network net, a Categorical distribution class, and a state
        2/ Compute the output 
            pdparams = net(state)
        3/ Construct and instance of an action prob distribution 
            pd = Categorical(logits=pdparams)
        4/ Use pd to sample an action, action = pd.sample()
        5/ Use pd and action to computer the action log prob
            log_prob = pd.log_prob(action)
    """
    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))  # convert numpy to tensor
        pdparam = self.forward(x) # Calculate the forward pass of the network
        pd = Categorical(logits=pdparam)  # probability distribution
        action = pd.sample()  # pi(a|s) in action via pd
        log_prob = pd.log_prob(action)  # log_prob of pi(a|s)
        self.log_probs.append(log_prob)  # store for training
        return action.item()

# Function take in the pi class and Adam opitimization to return the loss of each step
def train(pi, optimizer):
    # Inner gradient-ascent loop of Reinforce algorithms
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)  # the returns
    future_ret = 0.0

    # compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret

    rets = torch.tensor(rets) # Convert returen into tensor
    log_probs = torch.stack(pi.log_probs)
    loss = -log_probs * rets  # gradient terms; Negative for maximizing
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()  # Backprob for computing the gradients
    optimizer.step()  # gradient-ascent, updates the weights (the policy parameter)
    return loss

def main():
    env = gym.make("CartPole-v0") # Create environment
    in_dim = env.observation_space.shape[0] # way to take in input dimension in gym
    out_dim = env.action_space.n # way to output output dimension in gym
    pi = Pi(in_dim, out_dim)  # policy pi_theta for Reinforce. Return the thing in forward pass (weighs of the network to keep track)

    # Note: optimizer are algorithms or methods used to change the attributes of your network such as weights and learning rate in order to reduce the loss
    # Optimizer helps to get the result faster
    optimizer = optim.Adam(pi.parameters(), lr=0.01)

    # Train thru 300 episode
    for epi in range(300):
        state = env.reset() # Reset the state everytime

        # As the training progrees, the total reward per episode should increase towards 200
        # For every episode, run the time step of 200
        for t in range(200):  # the cartpole model max time step is 200
            action = pi.act(state) # take action at the current state
            state, reward, done, _ = env.step(action) # keep track of state, reward, and done or not
            pi.rewards.append(reward) # append the reward to sum the rewards later.
            env.render()
            if done:
                break

        loss = train(pi, optimizer)

        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0 # if the total reward is larger than 195, we know that the training is complete
        pi.onpolicy_reset()  # clear memory after training
        print(
            f"Episode {epi}, loss: {loss}, \ total_reward: {total_reward}, solved: {solved}"
        )

if __name__ == "__main__":
    main()
