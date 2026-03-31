from stable_baselines3 import DQN

from agents.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    name = "DQN"
    algo_cls = DQN
