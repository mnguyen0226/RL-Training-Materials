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
"""
from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1/ Initialize learning rate
gamma = 0.99


class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # set training mode

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))  # convert numpy to tensor
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam)  # probability distribution
        action = pd.sample()  # pi(a|s) in action via pd
        log_prob = pd.log_prob(action)  # log_prob of pi(a|s)
        self.log_probs.append(log_prob)  # store for training
        return action.item()


def train(pi, optimizer):
    # Inner gradient-ascent loop of Reinforce algo
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)  # the returns
    future_ret = 0.0

    # compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = -log_probs * rets  # gradient terms; Negative for maximizing
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()  # Backprob for computing the gradients
    optimizer.step()  # gradient-ascent, updates the weights
    return loss


def main():
    env = gym.make("CartPole-v0")
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)  # policy pi_theta for Reinforce
    optimizer = optim.Adam(pi.parameters(), lr=0.01)

    for epi in range(300):
        state = env.reset()
        for t in range(200):  # the cartpole model max time step is 200
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                break
        loss = train(pi, optimizer)  # train per episode

        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()  # clear memory after training
        print(
            f"Episode {epi}, loss: {loss}, \ total_reward: {total_reward}, solved: {solved}"
        )


if __name__ == "__main__":
    main()
