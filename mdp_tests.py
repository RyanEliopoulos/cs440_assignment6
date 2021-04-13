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


        
                


