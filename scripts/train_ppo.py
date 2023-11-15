import context

from dataclasses import dataclass

import torch
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from torch import nn

from src.logger import Logger
from src.envs.intersection import Intersection
import configs


class PPOAgent:
    def __init__(self, env, policy_model, value_model, clip_param=0.2):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.clip_param = clip_param
        self.horizon = 50

        self.saved_actions = []
        self.saved_log_probs = []
        self.rewards = []
        self.states = []

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

    def run_episode(self, max_steps: int, seed: int = None) -> tuple[dict, list[int]]:
        state = self.env.reset(seed)
        init_state = state.copy()
        actions = []
        for _ in range(max_steps):
            action = self.select_action(state)
            actions.append(action)
            state, reward = self.env.step(action)
            self.rewards.append(reward)

            if sum([len(i) for i in state['cars'].values()]) == 0:
                break

        return init_state, actions

    def select_action(self, state):
        state_tensor = self._convert_state(state)
        action_probs = self.policy_model(state_tensor)
        m = Categorical(action_probs)
        action = m.sample()

        self.saved_actions.append(action)
        self.saved_log_probs.append(m.log_prob(action))
        self.states.append(state_tensor)

        return action.item()

    def compute_returns(self, next_value, gamma):
        R = next_value
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def update_policy(self, optimizer_policy, optimizer_value, returns, old_log_probs, states, actions, clip_param, eps_clip):
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        old_log_probs = torch.cat(old_log_probs)

        for _ in range(clip_param):
            new_log_probs, state_values = [], []
            for state, action in zip(states, actions):
                action_probs = self.policy_model(state)
                m = Categorical(action_probs)

                new_log_probs.append(m.log_prob(action))
                state_values.append(self.value_model(state))

            new_log_probs = torch.cat(new_log_probs)

            state_values = torch.cat(state_values).squeeze()

            advantage = returns - state_values.detach()

            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantage

            policy_loss = -torch.min(surr1, surr2)
            value_loss = 0.5 * (state_values - returns).pow(2)

            optimizer_policy.zero_grad()
            policy_loss.mean().backward()
            optimizer_policy.step()

            optimizer_value.zero_grad()
            value_loss.mean().backward()
            optimizer_value.step()

            return (policy_loss.mean() + value_loss.mean()).item()

    def train(self, epochs, lr, gamma, logger=None):
        optimizer_policy = optim.Adam(self.policy_model.parameters(), lr=lr)
        optimizer_value = optim.Adam(self.value_model.parameters(), lr=lr)

        training_losses, avg_waiting_times = [], []
        for e in range(epochs):
            self.run_episode(200)

            next_value = self.value_model(self._convert_state(self.env.reset())).detach().item()
            returns = self.compute_returns(next_value, gamma)

            old_log_probs = self.saved_log_probs
            actions = torch.stack(self.saved_actions)
            states = torch.stack(self.states)

            loss = self.update_policy(optimizer_policy, optimizer_value, returns, old_log_probs, states, actions, 5, 0.2)
            avg_waiting_time = -sum(self.rewards) / len(self.rewards)  # Assuming rewards are negative waiting times
            training_losses.append(loss)
            avg_waiting_times.append(avg_waiting_time)

            self.saved_actions = []
            self.saved_log_probs = []
            self.rewards = []
            self.states = []

            if (e + 1) % 1 == 0 and logger is not None:
                logger.info(f'Epoch: {e + 1}, Loss: {loss}, Avg Waiting Time: {avg_waiting_time}')

        return training_losses, avg_waiting_times


@dataclass
class PPOConfig(configs.PolicyGradientConfig):
    lr: float = 1e-3


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, with_softmax: bool = True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, output_dim),
        )
        self.with_softmax = with_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        if self.with_softmax:
            x = nn.functional.softmax(x)
        return x


if __name__ == '__main__':
    logger = Logger(
        'PPO',
        configs.PathConfig().logs,
    )
    ppo_config = PPOConfig()
    logger.info(ppo_config)

    env = Intersection()
    policy_model = MLP((ppo_config.horizon + 1) * 4, 2)
    value_model = MLP((ppo_config.horizon + 1) * 4, 1, with_softmax=False)
    agent = PPOAgent(env, policy_model, value_model)

    with torch.no_grad():
        _, actions = agent.run_episode(ppo_config.steps_per_episode, 0)
        logger.info(f'Random actions: {actions}')
        env.reset(0)
        gif_path = configs.PathConfig().data / 'random_policy.gif'
        reward = env.render_to_gif(gif_path, actions)
        avg_waiting_time = -reward / len(actions)
        logger.info(f'Random intersection gif is saved at: {gif_path}. Avg time = {avg_waiting_time}')

    losses, time = agent.train(
        epochs=ppo_config.epochs,
        lr=ppo_config.lr,
        gamma=ppo_config.gamma,
        logger=logger
    )
    print(time)
    # Plot losses vs epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss over Epochs')
    ax.grid(True)
    plot_path = configs.PathConfig().data / 'ppo_training_loss_plot.png'
    fig.savefig(plot_path)
    logger.info(f'Loss plot is saved at: {plot_path}')
    plt.close(fig)

    # Plot losses vs epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Time')
    ax.set_title('Average waiting time over Epochs')
    ax.grid(True)
    plot_path = configs.PathConfig().data / 'ppo_time_plot.png'
    fig.savefig(plot_path)
    logger.info(f'Time plot is saved at: {plot_path}')
    plt.close(fig)

    # Save test results
    with torch.no_grad():
        _, actions = agent.run_episode(ppo_config.steps_per_episode, 0)
        logger.info(f'Learned actions: {actions}')
        env.reset(0)
        gif_path = configs.PathConfig().data / 'PPO_agent.gif'
        reward = env.render_to_gif(gif_path, actions)
        avg_waiting_time = -reward / len(actions)
        logger.info(f'Fine tuned intersection gif is saved at: {gif_path}. Avg time = {avg_waiting_time}')
