class BuyAndHoldAgent:
    """
    Baseline agent: buys all on day 1, holds until the end.
    This is the standard benchmark -- the RL agent needs to beat this to be useful.

    Pure policy class -- no I/O, no side effects.
    All run logic lives in runners/baseline_runner.py.
    """

    def act(self, step: int) -> int:
        return 4 if step == 0 else 2
