"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore


#################################################
# 1. Play with QLearningAgent
#################################################


agent_qlearn = QLearningAgent(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)

agent_qlearn_scheduling = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

agent_sarsa = SarsaAgent(
    learning_rate=0.5, epsilon=0.02, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(
    env: gym.Env, agent: QLearningAgent | SarsaAgent, t_max=int(1e4)
) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()
    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)
        next_s, r, done, _, _ = env.step(a)
        total_reward += r  # pyright: ignore
        if done:
            break
        # Train agent for state s
        agent.update(s, a, r, next_s)
        s = next_s
    return total_reward  # pyright: ignore

def plot_reward(agent, num_simulation=100):
    plot_rewards = []
    for sim in range(num_simulation):
        rewards = []
        for _ in range(3000):
            rewards.append(play_and_train(env, agent))
        if sim % 10 == 0:
            print(f"Simulation: {sim + 10}/{num_simulation}")
        plot_rewards.append(rewards)
    return np.array(plot_rewards)

plot_rewards_qlearn = plot_reward(agent_qlearn)
plot_rewards_qlearn_scheduling = plot_reward(agent_qlearn_scheduling)
plot_rewards_sarsa = plot_reward(agent_sarsa)


number_generation = 5

env = RecordVideo(
    env, video_folder="video", name_prefix="eval", episode_trigger=lambda x: True
)
env = RecordEpisodeStatistics(env, deque_size=number_generation)
for _ in range(number_generation):
    s, _ = env.reset()
    episode_over = False
    while not episode_over:
        action = agent_sarsa.get_best_action(s)  # replace with actual agent
        s, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated
env.close()

import matplotlib.pyplot as plt

plt.figure(figsize=(24, 12))
x = np.arange(1, 3001)
plt.plot(x, np.array(plot_rewards_qlearn.mean(axis=0)), label="Q-Learning")
plt.plot(x, np.array(plot_rewards_qlearn_scheduling.mean(axis=0)), label="Q-Learning with scheduling")
plt.plot(x, np.array(plot_rewards_sarsa.mean(axis=0)), label="SARSA")

plt.title("Comparison of different RL models", fontsize=20)
plt.xlabel("Iterations (log scale)", fontsize=16)
plt.ylabel("Rewards (average of 100 simulations)", fontsize=16)

plt.legend(fontsize=10)
plt.grid(True)

plt.xscale('log')
plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])

plt.savefig("report/results_exploitation.png", dpi=300, bbox_inches='tight')