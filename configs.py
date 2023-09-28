from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class PolicyGradientConfig:
    horizon: int = 50
    epochs: int = 200
    episodes_per_epoch: int = 30
    max_steps_per_episode: int = 50
    lr: float = 1e-3


@dataclass
class PathConfig:
    project: Path = Path(__file__).absolute().parent
    src: Path = project / 'src'
    data: Path = project / 'data'
    scripts: Path = project / 'scripts'
    tests: Path = project / 'tests'
    logs: Path = data / 'logs'

    def __post_init__(self):
        for i in vars(self):
            path = getattr(self, i)
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggerConfig:
    level: int | str = logging.INFO
    path = PathConfig().logs
