from collections import defaultdict
import random
import typing as t
import numpy as np


Action = int
State = int
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[int, t.DefaultDict[Action, float]]


class SarsaAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        legal_actions: t.List[Action],
    ):
        """
        SARSA  Agent

        You shoud not use directly self._qvalues, but instead of its getter/setter.
        """
        self.legal_actions = legal_actions
        self.epsilon = epsilon
        self._qvalues: QValues = defaultdict(lambda: defaultdict(int))
        self.learning_rate = learning_rate
        self.gamma = gamma

    def get_qvalue(self, state: State, action: Action) -> float:
        """
        Returns Q(state,action)
        """
        return self._qvalues[state][action]

    def set_qvalue(self, state: State, action: Action, value: float):
        """
        Sets the Qvalue for [state,action] to the given value
        """
        self._qvalues[state][action] = value

    def get_value(self, state: State) -> float:
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_a Q(s, a) over possible actions.
        """
        value = 0.0
        # BEGIN SOLUTION
        #
        # END SOLUTION
        return value

    def update(
        self, state: State, action: Action, reward: t.SupportsFloat, next_state: State
    ):
        """
        You should do your Q-Value update here (s'=next_state):
           TD_target(s') = R(s, a) + gamma * V(s')
           TD_error(s', a) = TD_target(s') - Q(s, a)
           Q_new(s, a) := Q(s, a) + alpha * TD_error(s', a)
        """
        # BEGIN SOLUTION
        td_target = reward + self.gamma * self.get_qvalue(
            next_state, self.get_action(next_state)
        )  # pyright: ignore
        td_error = td_target - self.get_qvalue(state, action)
        new_q_value = self.get_qvalue(state, action) + self.learning_rate * td_error
        self.set_qvalue(state, action, new_q_value)
        # END SOLUTION

    def get_best_action(self, state: State) -> Action:
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_q_values = [
            self.get_qvalue(state, action) for action in self.legal_actions
        ]
        index = np.argmax(possible_q_values)
        best_action = self.legal_actions[index]
        return best_action

    def get_action(self, state: State) -> Action:
        """
        Compute the action to take in the current state, including exploration.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.legal_actions)
        else:
            return self.get_best_action(state)
