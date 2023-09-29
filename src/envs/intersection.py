import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import imageio

from pathlib import Path

from src.envs.car import Car
from src.envs.env import Env


matplotlib.use('Agg')


class Intersection(Env):
    """
    An intersection can:
    1. Perceive cars location on the road (represented by grid distances)
    2. Control traffic light (0 for red, 1 for green)
    """

    def __init__(self, seed: int = None):
        self.cars, self.traffic_light = dict(), dict()
        self.reset(seed)

    def reset(self, seed: int = None):
        """
        Reset intersection to its initial state using NumPy for random number generation
        """
        if seed is not None:
            np.random.seed(seed)

        directions = ['north', 'east', 'south', 'west']
        self.cars = {
            k: [
                Car(i) for i in
                sorted(np.random.choice(range(1, 101), 70, replace=False))
            ]
            for k in directions
        }
        self.traffic_light = {'north': 1, 'east': 0, 'south': 1, 'west': 0}
        return self.get_state()

    def get_state(self):
        """
        Get the current state of the intersection.
        """
        state = {'cars': self.cars, 'traffic_light': self.traffic_light}
        return state

    def step(self, action: int):
        """
        Take an action to change the traffic light and return the new state.
        Action can be a dictionary specifying which direction to turn the light green.
        """

        # Update traffic light based on action
        action_to_traffic_light = {
            0: {'north': 1, 'east': 0, 'south': 1, 'west': 0},
            1: {'north': 0, 'east': 1, 'south': 0, 'west': 1},
        }
        self.traffic_light = action_to_traffic_light[action]

        # Simulate car movement based on the new traffic light state
        waiting_time = 0
        for direction, light in self.traffic_light.items():
            front_car_position = -1
            for car in self.cars[direction]:
                if car.position == 0 and light == 1:
                    self.cars[direction].pop(0)
                else:
                    if car.position - front_car_position > 1:
                        car.move()
                    else:
                        waiting_time += 1
                front_car_position = car.position

        # Compute reward
        reward = -waiting_time

        # Get new state after the step
        new_state = self.get_state()

        return new_state, reward

    def render_to_array(self, grid_size=100) -> np.ndarray:
        # Initialize a grid of zeros
        grid = np.zeros((grid_size, grid_size))

        # Draw roads (set value to 1)
        grid[grid_size // 2, :] = 1
        grid[:, grid_size // 2] = 1

        # Draw traffic lights
        for direction, light_status in self.traffic_light.items():
            match direction:
                case 'north':
                    row, col = grid_size // 2 - 1, grid_size // 2
                case 'south':
                    row, col = grid_size // 2 + 1, grid_size // 2
                case 'east':
                    row, col = grid_size // 2, grid_size // 2 + 1
                case 'west':
                    row, col = grid_size // 2, grid_size // 2 - 1
                case _:
                    raise ValueError("Invalid direction")

            grid[row, col] = 3 if light_status else 2  # 3 for green, 2 for red

        # Draw cars (set value to 4)
        for direction, cars in self.cars.items():
            for car in cars:
                match direction:
                    case 'north':
                        row = grid_size // 2 - car.position - 2
                        col = grid_size // 2
                    case 'south':
                        row = grid_size // 2 + car.position + 2
                        col = grid_size // 2
                    case 'east':
                        row = grid_size // 2
                        col = grid_size // 2 + car.position + 2
                    case 'west':
                        row = grid_size // 2
                        col = grid_size // 2 - car.position - 2
                    case _:
                        raise ValueError("Invalid direction")

                if 0 <= row < grid_size and 0 <= col < grid_size:
                    grid[row, col] = 4  # 4 for car

        return grid

    def _generate_img(self, ax, grid_size: int = 30):
        grid = self.render_to_array(grid_size)
        cm = mcolors.LinearSegmentedColormap.from_list(
            'traffic',
            [(1.0, 1.0, 1.0), (0.7, 0.7, 0.7), (1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0)],
            N=5
        )
        ax.imshow(grid, cmap=cm)
        ax.axis('off')

    def render_to_image(self, path: str | Path, grid_size: int = 30):
        fig, ax = plt.subplots()
        self._generate_img(ax, grid_size)
        ax.set_title('Intersection State')
        fig.savefig(path)
        plt.close(fig)

    def render_to_gif(self, path: str | Path, actions: list[int], grid_size: int = 30) -> int:
        images = []
        total_reward = 0
        for action in actions:
            fig, ax = plt.subplots()
            _, reward = self.step(action)  # Assuming step is defined
            total_reward += reward
            self._generate_img(ax, grid_size)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            plt.close(fig)

        imageio.mimsave(path, images, duration=0.5)
        return total_reward


