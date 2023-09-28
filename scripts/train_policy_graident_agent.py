import matplotlib.pyplot as plt

from src.logger import Logger
from src.agents.policy_gradient_agent import PolicyGradientAgent
from src.envs.intersection import Intersection
from src.models.policy_model import PolicyModel
import configs


if __name__ == '__main__':
    logger = Logger(
        'policy_gradient',
        configs.PathConfig.logs,
    )

    env = Intersection()
    model = PolicyModel((configs.PolicyGradientConfig.horizon + 1) * 4, 2)
    agent = PolicyGradientAgent(env, model, configs.PolicyGradientConfig.horizon)
    losses = agent.train(
        epochs=configs.PolicyGradientConfig.epochs,
        episodes_per_epoch=configs.PolicyGradientConfig.episodes_per_epoch,
        max_steps_per_episode=configs.PolicyGradientConfig.max_steps_per_episode,
        lr=configs.PolicyGradientConfig.lr,
        logger=logger
    )
    # Plot losses vs epochs
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.savefig(configs.PathConfig.data / 'training_loss_plot.png')  # Save the plot
    plt.show()

    # Save test results
    init_state, actions = agent.run_episode(configs.PolicyGradientConfig.max_steps_per_episode, 0)
    env.cars = init_state['cars']
    env.traffic_light = init_state['traffic_light']
    env.render_to_gif(configs.PathConfig.data / 'policy_gradient_agent.gif', actions)

