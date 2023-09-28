import matplotlib.pyplot as plt
import torch.cuda

from src.logger import Logger
from src.agents.policy_gradient_agent import PolicyGradientAgent
from src.envs.intersection import Intersection
from src.models.policy_model import PolicyModel
import configs


if __name__ == '__main__':
    logger = Logger(
        'policy_gradient',
        configs.PathConfig().logs,
    )
    logger.info(configs.PolicyGradientConfig())

    env = Intersection()
    model = PolicyModel((configs.PolicyGradientConfig().horizon + 1) * 4, 2)
    logger.info(f'Model:\n{model}')
    agent = PolicyGradientAgent(env, model, configs.PolicyGradientConfig().horizon)

    with torch.no_grad():
        _, actions = agent.run_episode(configs.PolicyGradientConfig().max_steps_per_episode, 0)
        logger.info(f'Random actions: {actions}')
        env.reset(0)
        gif_path = configs.PathConfig().data / 'random_policy.gif'
        reward = env.render_to_gif(gif_path, actions)
        avg_waiting_time = -reward / configs.PolicyGradientConfig().max_steps_per_episode
        logger.info(f'Random intersection gif is saved at: {gif_path}. Avg time = {avg_waiting_time}')

    losses = agent.train(
        epochs=configs.PolicyGradientConfig().epochs,
        episodes_per_epoch=configs.PolicyGradientConfig().episodes_per_epoch,
        max_steps_per_episode=configs.PolicyGradientConfig().max_steps_per_episode,
        lr=configs.PolicyGradientConfig().lr,
        logger=logger
    )
    # Plot losses vs epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss over Epochs')
    ax.grid(True)
    plot_path = configs.PathConfig().data / 'pg_training_loss_plot.png'
    fig.savefig(plot_path)
    logger.info(f'Loss plot is saved at: {plot_path}')
    plt.close(fig)

    # Save test results
    with torch.no_grad():
        _, actions = agent.run_episode(configs.PolicyGradientConfig().max_steps_per_episode, 0)
        logger.info(f'Learned actions: {actions}')
        env.reset(0)
        gif_path = configs.PathConfig().data / 'policy_gradient_agent.gif'
        reward = env.render_to_gif(gif_path, actions)
        avg_waiting_time = -reward / configs.PolicyGradientConfig().max_steps_per_episode
        logger.info(f'Fine tuned intersection gif is saved at: {gif_path}. Avg time = {avg_waiting_time}')
