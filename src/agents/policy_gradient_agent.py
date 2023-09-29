import torch
import torch.optim as optim

from src.envs.env import Env
from src.logger import Logger


class PolicyGradientAgent:
    def __init__(self, env: Env, model: torch.nn.Module, horizon: int):
        self.env = env
        self.model = model
        self.horizon = horizon
        self.saved_probs = []
        self.saved_rewards = []

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

    def select_action(self, state: dict) -> int:
        state_tensor = self._convert_state(state)
        action_probs = self.model(state_tensor).squeeze()
        action = torch.multinomial(action_probs, 1).item()
        self.saved_probs.append(torch.log(action_probs[action]))
        return action

    def run_episode(self, max_steps: int, seed: int = None) -> tuple[dict, list[int]]:
        state = self.env.reset(seed)
        init_state = state.copy()
        actions = []
        for _ in range(max_steps):
            action = self.select_action(state)
            actions.append(action)
            state, reward = self.env.step(action)
            self.saved_rewards.append(reward)

            if sum([len(i) for i in state['cars'].values()]) == 0:
                break

        return init_state, actions

    def train(
            self,
            epochs: int = 100,
            episodes_per_epoch: int = 3,
            max_steps_per_episode: int = 50,
            lr: float = 1e-3,
            gamma: float = 0.99,
            logger: Logger = None
    ) -> list[float]:
        training_losses = []
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for e in range(epochs):
            self.model.train()
            loss = torch.tensor(0.0)
            for _ in range(episodes_per_epoch):
                self.run_episode(max_steps_per_episode)

                # Calculate Monte Carlo returns
                current_return = 0
                returns = []
                for r in reversed(self.saved_rewards):
                    current_return = r + gamma * current_return
                    returns.insert(0, current_return)
                returns = torch.tensor(returns)

                # Normalize
                returns = (returns - returns.min()) / (returns.std() + 1e-9)

                # Calculate loss based on saved log probabilities and returns
                loss += -torch.sum(torch.stack(self.saved_probs) * returns)

                # Clear saved rewards and probabilities
                self.saved_probs = []
                self.saved_rewards = []

            # Update policy
            loss /= episodes_per_epoch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_losses.append(loss.item())
            if logger is not None:
                logger.info(f'Epoch: {e + 1}, training loss: {loss.item()}')
                # if loss.item() < 1e-9:
                #     break

        return training_losses

