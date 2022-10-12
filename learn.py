from environment import CliffBoxPushingBase
from collections import defaultdict
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Union
from tqdm import tqdm
import torch
import copy
from collections import deque
import ipdb

sync_freq = 50
loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.SmoothL1Loss()
learning_rate = 1e-3
device = torch.device("cuda")


class QAgent(object):
    def __init__(self, algorithm="sarsa"):
        supported_algorithms = ["sarsa", "qlearning", "dqn"]
        assert algorithm in supported_algorithms, f"Invalid algorithm. Must be one in {supported_algorithms}"
        self.algo = algorithm
        self.action_space = [1, 2, 3, 4]  # (up: 1, down: 2, left: 3, right: 4)

        # self.V = []
        if self.algo != "dqn":
            self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))
        else:
            self.mem_size = 1000
            self.batch_size = 50
            self.replay = deque(maxlen=self.mem_size)
            self.dqn_1 = torch.nn.Sequential(
                torch.nn.Linear(4, 150),
                torch.nn.ReLU(),
                torch.nn.Linear(150, 150),
                torch.nn.ReLU(),
                torch.nn.Linear(150, 4)
            ).to(device)
            self.dqn_2 = copy.deepcopy(self.dqn_1)
            self.dqn_2.load_state_dict(self.dqn_1.state_dict())
            self.optimizer = torch.optim.Adam(
                self.dqn_1.parameters(), lr=learning_rate)
            self.losses = []
            self.Q = None

        self.discount_factor = 0.99
        self.alpha = 0.5
        self.epsilon = 0.01

    # Epsilon greedy policy #
    def take_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            if self.algo != "dqn":
                action = self.action_space[np.argmax(self.Q[state])]
            else:
                action = self.action_space[np.argmax(self.Q)]
        return action

    def qlearning_algo(self, state, action, reward, next_state):
        # Q-Leaning updates by greedy action so it converges faster but can be more unstable
        q_next = np.max(self.Q[next_state][:])
        q_current = self.Q[state][action-1]
        return q_current + self.alpha * \
            (reward+self.discount_factor *
             q_next - q_current)

    def sarsa_algo(self, state, action, reward, next_state, policy_next_action):
        # SARSA updates by policy action so it is more robust and consistent but might not converge as well as Q learning
        q_next = self.Q[next_state][policy_next_action-1]
        q_current = self.Q[state][action-1]
        return q_current + self.alpha * \
            (reward+self.discount_factor *
             q_next - q_current)

    def train(self, state, action, next_state, reward):

        if self.algo == "qlearning":
            self.Q[state][action -
                          1] = self.qlearning_algo(state, action, reward, next_state)
        elif self.algo == "sarsa":
            policy_next_action = self.take_action(next_state)
            self.Q[state][action -
                          1] = self.sarsa_algo(state, action, reward, next_state, policy_next_action)
        elif self.algo == "dqn":
            state_input = torch.Tensor(
                state).view(1, -1).to(device)

            next_state_input = torch.Tensor(
                next_state).view(1, -1).to(device)
            qval = self.dqn_1(state_input)
            self.Q = qval.cpu().data.numpy()
            exp = (state_input, action-1, reward,
                   next_state_input, terminated)
            self.replay.append(exp)

            # updating q values by parameter learning
            if len(self.replay) > self.batch_size:
                minibatch = random.sample(self.replay, self.batch_size)

                state_batch = torch.cat(
                    [s1 for (s1, a, r, s2, d) in minibatch]).to(device)
                action_batch = torch.Tensor(
                    [a for (s1, a, r, s2, d) in minibatch]).to(device)
                reward_batch = torch.Tensor(
                    [r for (s1, a, r, s2, d) in minibatch]).to(device)
                next_state_batch = torch.cat(
                    [s2 for (s1, a, r, s2, d) in minibatch]).to(device)
                terminated_batch = torch.Tensor(
                    [d for (s1, a, r, s2, d) in minibatch]).to(device)

                Q1 = self.dqn_1(state_batch)

                with torch.no_grad():
                    Q2 = self.dqn_2(next_state_input)

                # Calculate loss from labels
                Y = reward_batch + self.discount_factor * \
                    ((1-terminated_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(
                    dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.losses.append(loss.item())
                self.optimizer.step()

        elif self.algo == "ppo":
            pass


def smooth_values(values: List[Union[int, float]], window: int = 50):
    return pd.DataFrame(values)[0].rolling(window).mean().dropna().values.tolist()


def save_qvalues(qdict: Dict, action_space: List, title="sarsa", output_dir="plots_and_qvalues"):
    df = pd.DataFrame(qdict).T
    df.columns = action_space
    df.reset_index(inplace=True)
    df.to_csv(f"{output_dir}/{title}_q_values.csv", index=None)


def plot_losses(total_losses: List[float], title="dqn", output_dir="plots_and_qvalues"):
    total_losses = smooth_values(total_losses)
    plt.figure(1, figsize=(12, 10))
    plt.title(f"{title} loss curve")
    plt.xlabel("No. Episodes")
    plt.ylabel("Cumulative Ep. Loss")
    plt.plot(range(len(total_losses)),
             total_losses, label='Model Performance')

    plt.savefig(
        f"{output_dir}/{title}_loss_curve.jpg")
    plt.show()


def plot_performance(total_rewards: List[float], total_episodes: List[int], title="sarsa", output_dir="plots_and_qvalues", benchmark=-388):
    total_rewards = smooth_values(total_rewards)
    total_episodes = smooth_values(total_episodes)
    plt.figure(1, figsize=(12, 10))
    plt.title(f"{title} reward curve")
    plt.xlabel("No. Episodes")
    plt.ylabel("Cumulative Ep. Reward")

    # Model reward curve
    plt.plot(range(len(total_rewards)),
             total_rewards, label='Model Performance')

    # Ideal Human reward curve
    plt.plot(range(len(total_rewards)),
             [benchmark]*len(total_rewards), label='Benchmark Performance')
    plt.legend(loc='lower right')

    plt.savefig(
        f"{output_dir}/{title}_{len(total_rewards)}ep_cum_rewards_vs_episode.jpg")
    plt.show()

    plt.figure(2, figsize=(12, 10))
    plt.title(f"{title} episode length curve")
    plt.xlabel("No. Episodes")
    plt.ylabel("Episode length")

    # Model episode curve
    plt.plot(range(len(total_episodes)),
             total_episodes, label='Model Performance')

    # Ideal episode length
    plt.plot(range(len(total_episodes)),
             [41]*len(total_episodes), label='Benchmark Performance')
    plt.legend(loc='lower right')

    plt.savefig(
        f"{output_dir}/{title}_{len(total_rewards)}ep_episode_length_vs_episode.jpg")
    plt.show()


if __name__ == '__main__':
    env = CliffBoxPushingBase()
    # you can implement other algorithms
    agent = QAgent(algorithm="dqn")
    # agent = QAgent(algorithm="sarsa")
    # agent = QAgent(algorithm="qlearning")
    total_rewards = []
    total_episodes = []
    total_losses = []
    time_step = 0
    num_iterations = 30000
    logger = tqdm(total=num_iterations, desc="episode")
    for i in range(num_iterations):
        logger.update(1)
        terminated = False
        rewards = []
        episode = []
        agent.losses = []
        idx = 0
        env.reset()

        while not terminated:
            idx += 1
            state = env.get_state()
            action = agent.take_action(state)
            reward, terminated, _ = env.step([action])
            next_state = env.get_state()
            agent.train(state, action, next_state, reward)

            if agent.algo == "dqn" and idx % sync_freq == 0:  # C
                agent.dqn_2.load_state_dict(agent.dqn_1.state_dict())

            rewards.append(reward)
            episode.append(action)
            logger.write(
                f'episode: {i}, step: {time_step}, actions: {action}, reward: {reward}')
            time_step += 1

        total_losses.append(agent.losses)
        total_rewards.append(sum(rewards))
        total_episodes.append(len(episode))
        # print(f'print the historical actions: {env.episode_actions}')

    output_dir = f"plots_and_qvalues/{agent.algo}"
    os.makedirs(output_dir, exist_ok=True)
    if agent.algo != "dqn":
        save_qvalues(agent.Q, agent.action_space,
                     title=agent.algo, output_dir=output_dir)
    else:
        plot_losses(total_losses, output_dir=output_dir)
    plot_performance(total_rewards, total_episodes, title=agent.algo,
                     output_dir=output_dir)
