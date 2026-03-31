from stable_baselines3 import PPO

from agents.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    name = "PPO"
    algo_cls = PPO
