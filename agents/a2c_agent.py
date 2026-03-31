from stable_baselines3 import A2C

from agents.base_agent import BaseAgent


class A2CAgent(BaseAgent):
    name = "A2C"
    algo_cls = A2C
