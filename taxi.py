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


agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)

agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

agent = SarsaAgent(
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


rewards = []
for i in range(10000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0

number_generation = 5

env = RecordVideo(
    env, video_folder="video", name_prefix="eval", episode_trigger=lambda x: True
)
env = RecordEpisodeStatistics(env, deque_size=number_generation)
for _ in range(number_generation):
    s, _ = env.reset()
    episode_over = False
    while not episode_over:
        action = agent.get_best_action(s)  # replace with actual agent
        s, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated
env.close()
