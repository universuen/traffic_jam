import context

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from itertools import count
import time
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import random
from src.envs.env import Env
from tqdm import tqdm

from src.envs.intersection import Intersection
from src.logger import Logger

import configs


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, action_dim),
            # nn.Softmax(1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class QLEARNINGAgent:
    def __init__(self, env: Env, horizon: int, state_dim, action_dim, memory_size=10000, batch_size=16, gamma=0.99,
                 lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.horizon = horizon

        # my code
        self.memory = deque(
            maxlen=memory_size)  # deque是一个双端队列，可以在队首或队尾插入或删除元素。在DQN算法中，使用deque实现经验池来存储之前的经验，因为它可以在队尾插入新的经验，并在队首删除最老的经验，从而保持经验池的大小不变。
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.steps = 0
        self.action_dim = 2

    def select_action(self, state, eps):
        if random.random() < eps:
            action = random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                action_prob = self.policy_net(state)
                action = action_prob.argmax().item()
        #     print('action_prob', action_prob)
        # print('action', action)
        return action

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def _convert_state(self, state: dict) -> torch.Tensor:
        directions = ['north', 'east', 'south', 'west']
        sub_tensors = []
        for direction in directions:
            sub_tensor = torch.zeros(self.horizon + 1)
            # Record car states
            for car in state['cars'][direction]:
                if car.position < self.horizon:
                    sub_tensor[car.position] = 1
            # Record traffic light state
            sub_tensor[-1] = state['traffic_light'][direction]
            sub_tensors.append(sub_tensor)
        return torch.cat(sub_tensors).view(1, -1)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))



        state_batch = torch.FloatTensor(torch.stack(batch[0])).squeeze().to(self.device)
        action_batch = torch.LongTensor(batch[1]).squeeze().to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).squeeze().to(self.device)
        next_state_batch = torch.FloatTensor(torch.stack(batch[3])).squeeze().to(self.device)

        # print('state_batch', state_batch.shape)
        # print('action_batch', action_batch.shape)
        # print('reward_batch', reward_batch.shape)
        # print('next_state_batch', next_state_batch.shape)

        m = nn.BatchNorm1d(204)
        m.to(self.device)

        reward_batch = (reward_batch - reward_batch.min()) / (reward_batch.max() - reward_batch.min() + 1e-9)


        q_values = self.policy_net(m(state_batch)).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_net(m(next_state_batch)).max(1)[0]

        expected_q_values = reward_batch + self.gamma * next_q_values
        # print('q_values', q_values)
        # print('expected_q_values', expected_q_values)

        loss = self.loss_fn(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        # print(loss)
        for name, param in self.policy_net.named_parameters():
            # print('-----------------------------')
            param.grad.data.clamp_(-1, 1)
            # print(name, param.grad)
        self.optimizer.step()

        self.steps += 1
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_dqn(self, env, agent, eps_start=1, eps_end=0.1, eps_decay=0.996, max_episodes=300, max_steps=200):
        eps = eps_start
        list_loss = []
        list_reward = []
        for episode in tqdm(range(max_episodes)):
            sum_tmp = []
            reward_list = []
            state = env.reset()

            for step in range(max_steps):
                state_tensor0 = self._convert_state(state)
                action = agent.select_action(state_tensor0, eps)

                next_state, reward = env.step(action)
                state_tensor1 = self._convert_state(next_state)
                agent.store_transition(state_tensor0, action, reward, state_tensor1)
                state = next_state
                loss = agent.train()
                sum_tmp.append(loss)
                reward_list.append(reward)

                if sum([len(i) for i in state['cars'].values()]) == 0:
                    break

            if episode % 50 == 0:
                agent.update_target()
            eps = max(eps * eps_decay, eps_end)
            sum_tmp2 = []
            for item in sum_tmp:
                if item is not None:
                    sum_tmp2.append(item)

            print('loss', sum(sum_tmp2) / (len(sum_tmp2) - 1))
            list_loss.append(sum(sum_tmp2) / (len(sum_tmp2) - 1))
            print('reward',  -sum(reward_list) / len(reward_list))
            list_reward.append(-sum(reward_list) / len(reward_list))

        x = [i for i in range(len(list_loss))]
        y1 = list_reward
        print('list_reward', list_reward)
        print('list_loss', list_loss)

        return list_loss, list_reward




if __name__ == "__main__":
    env = Intersection()
    agent = QLEARNINGAgent(env, 50, 204, 2)

    losses, time = agent.train_dqn(env, agent)
    print(time)
    # Plot losses vs epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss over Epochs')
    ax.grid(True)
    plot_path = configs.PathConfig().data / 'q_training_loss_plot.png'
    fig.savefig(plot_path)
    # logger.info(f'Loss plot is saved at: {plot_path}')
    plt.close(fig)

    # Plot losses vs epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Time')
    ax.set_title('Average waiting time over Epochs')
    ax.grid(True)
    plot_path = configs.PathConfig().data / 'q_time_plot.png'
    fig.savefig(plot_path)
    # logger.info(f'Time plot is saved at: {plot_path}')
    plt.close(fig)
