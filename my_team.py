# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
import time

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=1000):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
    
## STUDENT ADDITIONS START HERE
"""
strategy notes:

different profiles:

    3 options:
        -0 attackers
            -distractor spy strategy (both far from border)
                -> distractor approaches defenders and tries to lure them away from the spy

            - snag and deep search strat (one very near to border)
                -> one snag agent, one deep attack agent

        -1 attacker
                 -bait and camp
                -> one island patroller, one attack

        -2 attacker
            - guard and chaser
                -> guard stays around powerup
                -> chaser hunts down attacker nearest to border

    -defensive:
        one pellet guard 
        one chaser
        island patrol (if there is a large island, stay around here)
        flanking strategy
            -> if just one enemy on our side, and both defenders, then flank the opponent

    -offensive:
        return after certain threshold
        distractor method
        snagging strategy (eat quick dots near border when defenders are at their spawn)
        ignore defenders if powered up
        
heuristics

    crosspoint of two circles = position of enemy (wigth noise)
        -> sample this multiple times and average it
        
    start analysis:
        -> get maze layout
        -> get powerup locations
        -> get food dots

    defensive:
        -enemy bounties (how much food dots they have eaten)
        -enemy distance
        -enemy distance from powerup
        -distance from powerup
        -is enemy powered up?
        -enemy distance from their home
        -food islands distance (island = neighboring food dots)
        -food island size

    offensive:
        -food islands distance (island = neighboring food dots)
        -food island size
        -enemy distance
        -distance from home
        -is powered up?
        -distance from powerup
        -what is my bounty?
        -remaining powerup time
"""

class Agent:
    """
    An agent must define a get_action method, but may also define the
    following methods which will be called if they exist:

    def register_initial_state(self, state): # inspects the starting state
    """

    def __init__(self, index=0):
        self.index = index

    def get_action(self, state):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        util.raise_not_defined()

class ValueEstimationAgent(Agent):
    """
      Abstract agent which assigns values to (state, action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s, a)
      policy(s) = arg_max_{a in actions} Q(s, a)

      Both ValueIterationAgent and QLearningAgent inherit
      from this agent. While a ValueIterationAgent has
      a model of the environment via a MarkovDecisionProcess
      (see mdp.py) that is used to estimate Q-Values before
      ever actually acting, the QLearningAgent estimates
      Q-Values while acting in the environment.
    """

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, num_training = 10):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.num_training = int(num_training)

    ####################################
    #    Override These Functions      #
    ####################################
    def get_q_value(self, state, action):
        """
        Should return Q(state, action)
        """
        util.raise_not_defined()

    def get_value(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s, a)
        """
        util.raise_not_defined()

    def get_policy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with get_action
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s, a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        util.raise_not_defined()

    def get_action(self, state):
        """
        state: can call state.get_legal_actions()
        Choose an action and return it.
        """
        util.raise_not_defined()

class ReinforcementAgent(ValueEstimationAgent):
    """
      Abstract Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        What you need to know:
                    - The environment will call
                      observe_transition(state, action, next_state, delta_reward),
                      which will call update(state, action, next_state, delta_reward)
                      which you should override.
        - Use self.get_legal_actions(state) to know which actions
                      are available in a state
    """
    ####################################
    #    Override These Functions      #
    ####################################

    def update(self, state, action, next_state, reward):
        """
                This class will call this function, which you write, after
                observing a transition and reward
        """
        util.raise_not_defined()

    ####################################
    #    Read These Functions          #
    ####################################

    def get_legal_actions(self, state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.action_fn(state)

    def observe_transition(self, state, action, next_state, delta_reward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episode_rewards += delta_reward
        self.update(state, action, next_state, delta_reward)

    def start_episode(self):
        """
          Called by environment when new episode is starting
        """
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0

    def stop_episode(self):
        """
          Called by environment when episode is done
        """
        if self.episodes_so_far < self.num_training:
            self.accum_train_rewards += self.episode_rewards
        else:
            self.accum_test_rewards += self.episode_rewards
        self.episodes_so_far += 1
        if self.episodes_so_far >= self.num_training:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def is_in_training(self):
        return self.episodes_so_far < self.num_training

    def is_in_testing(self):
        return not self.is_in_training()

    def __init__(self, action_fn = None, num_training=100, epsilon=0.5, alpha=0.5, gamma=1):
        """
        action_fn: Function which takes a state and returns the list of legal actions

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        if action_fn == None:
            action_fn = lambda state: state.get_legal_actions()
        self.action_fn = action_fn
        self.episodes_so_far = 0
        self.accum_train_rewards = 0.0
        self.accum_test_rewards = 0.0
        self.num_training = int(num_training)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)

    ################################
    # Controls needed for Crawler  #
    ################################
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_learning_rate(self, alpha):
        self.alpha = alpha

    def set_discount(self, discount):
        self.discount = discount

    def do_action(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.last_state = state
        self.last_action = action

    ###################
    # Pacman Specific #
    ###################
    def observation_function(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.last_state is None:
            reward = state.get_score() - self.last_state.get_score()
            self.observe_transition(self.last_state, self.last_action, state, reward)
        return state

    def register_initial_state(self, state):
        self.start_episode()
        if self.episodes_so_far == 0:
            print('Beginning %d episodes of Training' % (self.num_training))

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        delta_reward = state.get_score() - self.last_state.get_score()
        self.observe_transition(self.last_state, self.last_action, state, delta_reward)
        self.stop_episode()

        # Make sure we have this var
        if not 'episode_start_time' in self.__dict__:
            self.episode_start_time = time.time()
        if not 'last_window_accum_rewards' in self.__dict__:
            self.last_window_accum_rewards = 0.0
        self.last_window_accum_rewards += state.get_score()

        NUM_EPS_UPDATE = 100
        if self.episodes_so_far % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            window_avg = self.last_window_accum_rewards / float(NUM_EPS_UPDATE)
            if self.episodes_so_far <= self.num_training:
                train_avg = self.accum_train_rewards / float(self.episodes_so_far)
                print('\tCompleted %d out of %d training episodes' % (
                       self.episodes_so_far, self.num_training))
                print('\tAverage Rewards over all training: %.2f' % (
                        train_avg))
            else:
                test_avg = float(self.accum_test_rewards) / (self.episodes_so_far - self.num_training)
                print('\tCompleted %d test episodes' % (self.episodes_so_far - self.num_training))
                print('\tAverage Rewards over testing: %.2f' % test_avg)
            print('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE, window_avg))
            print('\tEpisode took %.2f seconds' % (time.time() - self.episode_start_time))
            self.last_window_accum_rewards = 0.0
            self.episode_start_time = time.time()

        if self.episodes_so_far == self.num_training:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))

class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)


        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter() # dictionary which returns 0 if you look up a key that hasn't been assigned yet


    def get_q_value(self, state, action):
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)]




    def compute_value_from_q_values(self, state):
        """
          Returns max_action Q(state, action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        value = float ("-inf") # values can be negative, so initialize at -inf
        actions = self.get_legal_actions(state)
        if not actions:
            return 0.0 # default value
       
        for action in actions:
            value = max(value, (self.get_q_value(state, action)))
        return value


    def compute_action_from_q_values(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.get_legal_actions(state)


        max_q = self.compute_value_from_q_values(state)
        best_actions = []


        # make a list of all of the actions that, combined with the state, have the maximum q-value
        # then choose randomly from that list
        for action in actions:
            if self.get_q_value(state, action) == max_q:
                best_actions.append(action)
       
        if not best_actions:
            return None


        return random.choice(best_actions)




    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.


          HINT: You might want to use util.flip_coin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.get_legal_actions(state)
        action = None
        "*** YOUR CODE HERE ***"


        # with a chance of 1 - epsilon, the computed best action will be selected
        # otherwise, choose random from all the legal actions
        if util.flip_coin(1 - self.epsilon):
            action = self.compute_action_from_q_values(state)
        else:
            action = random.choice(legal_actions)
       
        return action
       


    def update(self, state, action, next_state, reward):
        "*** YOUR CODE HERE ***"
        old_q_value = self.get_q_value(state, action)
        next_max_q = self.compute_value_from_q_values(next_state)
        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.discount * next_max_q)


        # update the q_values dictionary
        self.q_values[(state, action)] = new_q_value
       


    def get_policy(self, state):
        return self.compute_action_from_q_values(state)


    def get_value(self, state):
        return self.compute_value_from_q_values(state)

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def get_action(self, state):
        """
        Simply calls the get_action method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.get_action(self, state)
        self.do_action(state, action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite get_q_value
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.feat_extractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()


    def get_weights(self):
        return self.weights


    def get_q_value(self, state, action):
        """
          Should return Q(state, action) = w * feature_vector
          where * is the dot_product operator
        """
        "*** YOUR CODE HERE ***"
        # Christmas present from Arno
        weights = self.get_weights()
        features = self.feat_extractor.get_features(state, action)


        # Q(s, a) = w_1 f_1(s, a) + w_2 f_2(s, a) + ... + w_n f_n(s, a)
        dot_product = 0
        for feature_name, feature_value in features.items():
            dot_product += weights[feature_name] * feature_value
        return dot_product


    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        weights = self.get_weights()
        features = self.feat_extractor.get_features(state, action)


        # compute max q-value of all state-acion pairs of a state
        def max_q_value(state):
            actions = self.get_legal_actions(state)
            current_q_value = float("-inf") # initialize at -inf, because values can be negative


            for action in actions:
                q_value = self.get_q_value(state, action)
                if q_value > current_q_value:
                    current_q_value = q_value

            if current_q_value == float("-inf"):
                current_q_value = 0 # default value
            return current_q_value
       
        difference = (reward + self.discount * max_q_value(next_state)) - self.get_q_value(state, action)


        # update weight for each feature of the state-action pair
        for feature_name, feature_value in features.items():
            old_weight = weights[feature_name]
            new_weight = old_weight + self.alpha * difference * feature_value


            self.weights[feature_name] = new_weight


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)


        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

class SmartFridgeAgent(CaptureAgent):
        
    def get_features(self, game_state, action):
        features = util.Counter()

        enemiesList = self.get_opponents()
        oponentConfig = self.get_current_observation()
        teamCapsules = self.get_capsules_you_are_defending()
        totalCapsules = len(teamCapsules)*2
        

        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)