import pprint
import random
from math import log, sqrt
random.seed(0)

TwoXTwoMDP = {
    # Key (state id): Value (adjacent state via action L,R,U,D)
    'stategraph': { 1: [1,4,2,1], #Connections in order L,R,U,D
                     2: [2,3,2,1],
                     3: None,
                     4: None},
    # Probability of traversing each edge type (L,R,U,D) given each action
    'paction' : { 'L': [.8, 0, .1, .1],
                  'R': [0, .8, .1, .1],
                  'U': [.1, .1, .8, 0],
                  'D': [.1, .1, 0, .8]},

    'actions' : ['L','R','U','D']
}


FourXThreeMDP = {



    # Key (state id): Value (adjacent state via action L,R,U,D)
    'stategraph': { (1,1): [(1,1),(2,1),(1,2),(1,1)], #Connections in order L,R,U,D
                    (1,2): [(1,2),(1,2),(1,3),(1,1)],
                    (1,3): [(1,3),(2,3),(1,3),(1,2)],
                    (2,1): [(1,1),(3,1),(2,1),(2,1)],
                    (2,2): None,
                    (2,3): [(1,3),(3,3),(2,3),(2,3)], 
                    (3,1): [(2,1),(4,1),(3,2),(3,1)],
                    (3,2): [(3,2),(4,2),(3,3),(3,1)],
                    (3,3): [(2,3),(4,3),(3,3),(3,2)], 
                    (4,1): [(3,1),(4,1),(4,2),(4,1)],
                    (4,2): None,
                    (4,3): None },

    # Probability of traversing each edge type (L,R,U,D) given each action
    'paction' : { 'L': [.8, 0, .1, .1],
                  'R': [0, .8, .1, .1],
                  'U': [.1, .1, .8, 0],
                  'D': [.1, .1, 0, .8]},

    'actions' : ['L','R','U','D']
    
}


def maxutil(current_state: int, mdp: dict, state_utils: dict) -> float:
    """
        Returns the utility value of the optimal move

    :param current_state:
    :param mdp:
    :param state_utils:
    :return:
    """
    successors = mdp['stategraph'][current_state]
    # Return zero if current_state is a terminal state
    #print(f"Successor states: {successors}")
    if successors is None:
        return 0

    # Evaluating each action
    utilmax = None
    for action in mdp['actions']:
        u = 0

        # Scoring each possible outcome state
        for direction, prob in enumerate(mdp['paction'][action]):
            target_state = successors[direction]
            #print(f"Target state: {target_state}")
            u += prob * state_utils[target_state]

        if utilmax is None or (u > utilmax):
            utilmax = u
            bestaction = action

    return utilmax


def value_iteration(mdp, gamma, r_fn, quiet=True, delta=.0001, n=-1):
    """
    __ Part 1: Implement this __

    Perform Value Iteration:
    
     mdp - A Markov Decision Process represented as a dictionary with keys:
         'stategraph' : a map between states and transition vectors 
                        (next states reachable via a the action with 'index' i)
         'paction'    : a map between intended action and a distribution overal
                         *actual* actions, the actual action vector should have the
                         same length as the transition vectors in the stategraph, and
                         the sum of this vector should be 1.0
         'actions'    : the 'name' of each action in the action vector.

     gamma - a discount rate
     r_fn  - a reward function that takes a state and returns an immediate reward.
     quiet - if True, supress all output in this function
     delta - a stopping criteria: stops when utility changes by <= delta
     n     - a stopping criteria: stops when n iterations have occurred 
              (if n==-1, only the delta test applies)
     
     the stopping criteria for value iteration should have the following semantics:


       if (utility_change <= delta or (n > 0 and iterations == n)) then stop

     returns:
      a map of {state -> utilities}
    """

    # Initial utility of each state is the reward function
    state_utilities = dict()
    for state in mdp['stategraph'].keys():
        state_utilities[state] = r_fn(state)

    # Preparing for iteration
    iterated_utilities = dict(state_utilities)  # store for the "simultaneous" utility updates
    if n == -1:
        i = -2
    else:
        i = 0

    while i < n:
        iteration_improvement = 0
        for state in mdp['stategraph']:
            iterated_utilities[state] = r_fn(state) + (gamma * maxutil(state, mdp, state_utilities))
            if abs(iterated_utilities[state] - state_utilities[state]) > iteration_improvement:
                iteration_improvement = abs(iterated_utilities[state] - state_utilities[state])

        if iteration_improvement <= delta:
            break

        # Preparing for next loop
        state_utilities = dict(iterated_utilities)  # dict call ensures copy instead of reference
        if n != -1:
            i += 1

    if not quiet:
        print(f'There were {i} iterations')

    return state_utilities


def policy_evaluate(mdp, r_fn, gamma, policy, policy_utilities, viterations) -> dict:
    """
        Utility of each state depends on:
                                            1). the policy, informing the course of action
                                            2). the potential outcomes of each prescribed action

        Want to return a {state: utility} map
    """

    # Updating state utility values
    updated_utils = dict()
    for x in range(viterations):
        for state in mdp['stategraph'].keys():
            #print(f'This state now: {state}')
            successors = mdp['stategraph'][state]  # possible destination states
            if not successors:
                # No where to go. Utility is static
                updated_utils[state] = r_fn(state)
                continue

            # Tallying utility of the prescribed action
            new_u = 0
            action = policy[state]
            for direction_index, likelihood in enumerate(mdp['paction'][action]):
                target_state = mdp['stategraph'][state][direction_index]
                new_u += likelihood * policy_utilities[target_state]

            updated_utils[state] = r_fn(state) + (gamma * new_u)
        policy_utilities = updated_utils

    return updated_utils


def bestmove(state: tuple, mdp: dict, policy_utilities: dict):
    """
        Returns a tuple (utility of best move, best move) for the current state if there are moves to be made.
        Otherwise, returns (None, None)
    """

    successors = mdp['stategraph'][state]
    if not successors:
        return None, None

    best_action = None
    max_u = None
    # Evaluating each action
    for action in mdp['actions']:
        u = 0
        for direction, likelihood in enumerate(mdp['paction'][action]):
            target_state = mdp['stategraph'][state][direction]
            u += likelihood * policy_utilities[target_state]

        if max_u is None or u > max_u:
            max_u = u
            best_action = action

    return max_u, best_action


def policy_iteration(mdp, gamma, r_fn, policy, quiet=True, viterations=5):
    """
    __ Part 2: Implement this __

    Perform Policy Iteration:
    
     mdp - A Markov Decision Process represented as a dictionary with keys:
         'stategraph' : a map between states and transition vectors 
                        (next states reachable via a the action with 'index' i)
         'paction'    : a map between intended action and a distribution overal
                         *actual* actions, the actual action vector should have the
                         same length as the transition vectors in the stategraph, and
                         the sum of this vector should be 1.0
         'actions'    : the 'name' of each action in the action vector.

     gamma - a discount rate
     r_fn  - a reward function that maps states -> immediate rewards
     quiet - if True, supress all output in this function
     vit   - the number of iterations for the value update step
     n     - the number of iterations for policy update

     the stopping criteria for policy iteration should have the following semantics:

       if (not policy_changed or (n > 0 and iterations == n)) then stop

     returns:
      a map of {state -> actions}
    """




    """
        We just make up a default policy (always go up!) and then use that as the basis to improve against?
        ??????????????
        
       

             
        
        while True:
            state_utilities = PolicyEvaluate(pi, 
            # Get {'state': 'utility'} dict for each state based on the policy?
    """

    # set initial utility values to 0
    policy_utilities = dict()
    for state in mdp['stategraph'].keys():
        policy_utilities[state] = 0

    # Beginning improvement loop
    unchanged = False
    while not unchanged:
        policy_utilities = policy_evaluate(mdp, r_fn, gamma, policy, policy_utilities, viterations)
        #print(f"policy utilities: {policy_utilities}")
        unchanged = True
        for state in mdp['stategraph']:
            # Determining the best action for this state
            max_u, best_action = bestmove(state, mdp, policy_utilities)
            if max_u is None:  # No move to make
                continue

            if max_u > (policy_utilities[state] - r_fn(state)):
                old_val = policy_utilities[state]
                print(f'Updated {state}. old val: {old_val}, new val: {max_u}')
                old_action = policy[state]
                print(f'From action {old_action} to new action {best_action}')
                policy[state] = best_action
                unchanged = False

    return policy_utilities


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['policy','value'])
    parser.add_argument('environment', choices=['4x3', '2x2'])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)
    
    args = parser.parse_args()
    
    if args.environment == '4x3':
        env = FourXThreeMDP
        gamma = args.gamma
        rfn = lambda s: {(4,2):-1, (4,3):1}.get(s, -0.04)
        pi = {s:'L' for s in env['stategraph']}
        s0 = (1,1) # start state
    elif args.environment == '2x2':
        env = TwoXTwoMDP
        gamma = args.gamma
        # rfn is a reference to the function that performs the indexing
        # look at the documentation (for python's built-in list) for info
        rfn = [None, -0.04, -0.04, 1, -1].__getitem__
        pi = {1: 'R', 2:'D'}
        s0 = 1 # start state

    if args.method == 'policy':
        policy_iteration(env, gamma, rfn, pi, quiet=args.quiet)
    elif args.method == 'value':
        value_iteration(env, gamma, rfn, quiet=args.quiet)
