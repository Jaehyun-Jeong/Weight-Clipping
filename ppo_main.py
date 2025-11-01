import argparse
from datetime import datetime
import gymnasium as gym
import numpy as np
from ppo import Agent
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()  # Tensorboard


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99, help="discount")
    parser.add_argument("--alpha", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="gae lambda")
    parser.add_argument("--policy-clip", type=float, default=0.2, help="policy clip")
    parser.add_argument("--critic-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--norm-advantage", type=bool, default=True)
    parser.add_argument("--clip-value-loss", type=bool, default=True)
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--n", type=int, default=20, help="update frequency")
    parser.add_argument("--optimizer", type=str, default='Adam')
    parser.add_argument("--total-timsteps", type=int, default=1_000_000)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    print(0)
    env = gym.make("HalfCheetah-v5")
    print(1)
    N = args.n
    batch_size = args.batch_size
    n_epochs = args.epochs
    alpha = args.alpha
    num_iteration = args.total_timesteps // batch_size

    agent = Agent(
        n_actions=1,
        gamma=args.gamma,
        optimizer_name=args.optimizer,
        alpha=args.alpha,
        gae_lambda=args.gae_lambda,
        policy_clip=args.policy_clip,
        critic_coef=args.critic_coef,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        input_dims=env.observation_space.shape,
        norm_advantage=args.norm_advantage,
        clip_value_loss=args.clip_value_loss,
    )
    print(3)

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    while n_steps < args.total_timesteps:
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:

                # Annealing
                frac = 1.0 - (learn_iters - 1.0) / num_iterations
                lrnow = frac * args.alpha
                agent.actor.optimizer.param_group[0]["lr"] = lrnow
                agent.critic.optimizer.param_group[0]["lr"] = lrnow

                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            "episode",
            i,
            "score %.1f" % score,
            "avg score %.1f" % avg_score,
            "time_steps",
            n_steps,
            "learning_steps",
            learn_iters,
        )

    x = [i + 1 for i in range(len(score_history))]
    # plot_learning_curve(x, score_history, figure_file)
