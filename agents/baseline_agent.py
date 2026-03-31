class BuyAndHoldAgent:
    """
    Baseline agent: buys all on day 1, holds until the end.
    This is the standard benchmark — the RL agent needs to beat this to be useful.

    Pure policy class — no I/O, no side effects.
    All run logic lives in runners/baseline_runner.py.
    """

    def act(self, step: int) -> int:
        # Action 4 = buy all on first step, action 2 = hold every step after
        return 4 if step == 0 else 2