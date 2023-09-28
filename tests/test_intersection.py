from src.envs.intersection import Intersection
from src.envs.car import Car

from random import randint


# Initialize intersection
intersection = Intersection()
intersection.render_to_image('test.png')
actions = [randint(0, 1) for _ in range(100)]
intersection.render_to_gif('test.gif', actions)
