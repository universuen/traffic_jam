import matplotlib.pyplot as plt
import torch.cuda

from src.logger import Logger
from src.agents.reinforce_agent import REINFORCEAgent
from src.envs.intersection import Intersection
from src.models.policy_model import PolicyModel
import configs


if __name__ == '__main__':
    logger = Logger(
        'REINFORCE',
        configs.PathConfig().logs,
    )
    reinforce_config = configs.PolicyGradientConfig()
    logger.info(reinforce_config)

    env = Intersection()
    model = PolicyModel((reinforce_config.horizon + 1) * 4, 2)
    logger.info(f'Model:\n{model}')
    agent = REINFORCEAgent(env, model, reinforce_config.horizon)

    with torch.no_grad():
        _, actions = agent.run_episode(reinforce_config.steps_per_episode, 0)
        logger.info(f'Random actions: {actions}')
        env.reset(0)
        gif_path = configs.PathConfig().data / 'random_policy.gif'
        reward = env.render_to_gif(gif_path, actions)
        avg_waiting_time = -reward / len(actions)
        logger.info(f'Random intersection gif is saved at: {gif_path}. Avg time = {avg_waiting_time}')

    losses, time = agent.train(
        epochs=reinforce_config.epochs,
        episodes_per_epoch=reinforce_config.episodes_per_epoch,
        steps_per_episode=reinforce_config.steps_per_episode,
        lr=reinforce_config.lr,
        gamma=reinforce_config.gamma,
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
    plot_path = configs.PathConfig().data / 'reinforce_training_loss_plot.png'
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
    plot_path = configs.PathConfig().data / 'reinforce_time_plot.png'
    fig.savefig(plot_path)
    logger.info(f'Time plot is saved at: {plot_path}')
    plt.close(fig)

    # Save test results
    with torch.no_grad():
        _, actions = agent.run_episode(reinforce_config.steps_per_episode, 0)
        logger.info(f'Learned actions: {actions}')
        env.reset(0)
        gif_path = configs.PathConfig().data / 'REINFORCE_agent.gif'
        reward = env.render_to_gif(gif_path, actions)
        avg_waiting_time = -reward / len(actions)
        logger.info(f'Fine tuned intersection gif is saved at: {gif_path}. Avg time = {avg_waiting_time}')
