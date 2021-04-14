import re
import unittest
from mdp import *
import subprocess
#import timeout_decorator
import time


class MDPTestCase(unittest.TestCase):
    def test_2x2(self):
        rfn = [None, -0.04, -0.04, 1, -1].__getitem__
        d = value_iteration(TwoXTwoMDP, 1.0, rfn, True, n=1)
        self.assertAlmostEqual(d[1], -0.08, 3, 'After 1 iteration, expected -0.08 in state 1 of 2x2 environment')
        self.assertAlmostEqual(d[2], .752, 3, 'After 1 iteration, expected .752 in state 2 of 2x2 environment')

    def test_4x3(self):
        rfn = lambda s: {(4, 2): -1, (4, 3): 1}.get(s, -0.04)

        # Iteration 1
        d = value_iteration(FourXThreeMDP, 1.0, rfn, True, n=1)
        """
            After 1 iteration, expected Z in state (x, y) of 4x3 environment
        """
        outcomes = [
            -.08,  # (1, 1)
            -.08,  # (1, 2)
            -.08,
            -.08,  # (2, 1)
            -.04,
            -.08,
            -.08,  # (3, 1)
            -.08,
            .752,
            -.08,  # (4, 1)
            -1,
            1
        ]
        for i, state in enumerate(FourXThreeMDP['stategraph'].keys()):
            expected_outcome = outcomes[i]
            self.assertAlmostEqual(d[state], expected_outcome, 3, f'After 1 iteration, expected {expected_outcome} in state {state}')

        # Iteration 2
        outcomes2 = [
            -.12,  # (1, 1)
            -.12,
            -.12,
            -.12,  # (2, 1)
            -.04,
            .5456,
            -.12,  # (3, 1)
            .4536,
            .8272,
            -.12,  # (4, 1)
            -1,
            1
        ]

        d2 = value_iteration(FourXThreeMDP, 1.0, rfn, True, n=2)
        for i, state in enumerate(FourXThreeMDP['stategraph'].keys()):
            expected_outcome = outcomes[i]
            self.assertAlmostEqual(d[state], expected_outcome, 4, f'After 1 iteration, expected {expected_outcome} in state {state}')

    def test_policy_2x2(self):
        env = FourXThreeMDP
        rfn = lambda s: {(4, 2): -1, (4, 3): 1}.get(s, -0.04)
        pi = {s:'L' for s in env['stategraph']}
        d = policy_iteration(FourXThreeMDP, 1.0, rfn, pi, True, viterations=1)

        print(d)