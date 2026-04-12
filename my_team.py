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
from collections import Counter as CountList


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='SmartFridgeAgent', second='SmartFridgeAgent', num_training=0):
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
        #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

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

## functions for creating features  
def get_neighbor_data(x, y, data_matrix, neighbor_selection):
    """
    Given a position and a matrix, returns a tuple of lists: (true_data, false_data).
        \n true_data is a list of neighboring positions that return true in the matrix
        \n false_data is a list of neighboring positions that return false in the matrix
        \n This is an "abstract method" for `get_neighbor_walls` and `get_neighbor_food`. 
    """
    true_data = []
    false_data = []
    for dx, dy in neighbor_selection:
        neighbor = data_matrix[x+dx][y+dy]
        if neighbor:
            true_data.append((x+dx, y+dy))
        else:
            false_data.append((x+dx, y+dy))
    return true_data, false_data

def get_neighbor_walls(agent,x,y):
    """
    returns a tuple of lists: (walls, non_walls)
    \n walls is a list of positions that contain a wall
    \n non_walls is a list of positions that do not contain a wall
    """
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    return get_neighbor_data(x,y,agent.walls,neighbors)

def get_prev_positions(agent):
    """
    returns a list of the 12 most recent positions of the agent by looking at self.observation_history.
    """
    previous_positions = [observation.get_agent_position(agent.index)for observation in agent.observation_history[-12:]]
    return previous_positions
    
def get_bad_positions(previous_positions):
    """
    returns a list of positions which occur 6 times or more in a given list of positions.
    """
    uniquePositions = CountList(previous_positions).keys()
    bad_positions = [position for position in uniquePositions.mapping if uniquePositions.mapping.get(position) >= 6]
            
    return bad_positions

def get_dead_ends(agent, game_state):
        """
        returns a list of tuples (x,y) for all dead end cells. A dead end cell is defined as any cell with exactly 3 neighboring walls.
        """
        dead_ends = []

        for x in range(agent.walls.width):
            for y in range(agent.walls.height):

                if game_state.data.layout.walls[x][y]:
                    continue
                
                wall_neighbors, empty_neighbors = get_neighbor_walls(agent,x,y)
                if len(wall_neighbors) == 3:
                    dead_ends.append((x,y))
        return dead_ends

def breadth_first_search(start, stopcondition, data_matrix, neighbor_selection, return_visited, expand_into_true):
    """
    A generalized BFS algorithm to used by `BFS_food` and `BFS_walls`.
    \n `start`: the starting position for the BFS search.
    \n `Stopcondition`: a callable function that should return True or False. Determines when to stop early.
    \n `data_matrix`: matrix of bools in which the BFS algorithm will search
    \n `neighbor_selection`: list of tuples (x,y) that are used for calculating neighboring positions.
    \n `return_visited`: Boolean that determines if the Visisted list will be returned if the agenda is empty.
    \n `expand_into_true`: Boolean that detemines wether the BFS algorithm will expand into True values or False values
    """
    agenda = util.Queue()
    init_cell = start
    agenda.push([init_cell,[]])
    Visited = set([init_cell])

    while True:
        if agenda.is_empty():
            return Visited if return_visited else []

        current_state = agenda.pop()
        current_path = current_state[1]
        current_cell = current_state[0]

        true_list, false_list = get_neighbor_data(current_cell[0],current_cell[1], data_matrix, neighbor_selection)
        expand_list = true_list if expand_into_true else false_list

        if stopcondition(true_list):
            #self.debug_draw(current_cell,color=(0.9,0.2,0.2))
            return current_path

        for next_cell in expand_list:
            if next_cell not in Visited:
                Visited.add(next_cell)
                #self.debug_draw(current_cell,color=(0.5,0.8,0.3))
                agenda.push([next_cell, current_path + [current_cell]])

def BFS_walls(agent,start):
    def stopcondition(walls):
        return len(walls) < 2

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    return breadth_first_search(start, stopcondition=stopcondition,
                                data_matrix=agent.walls,
                                neighbor_selection=neighbors,
                                expand_into_true=False,
                                return_visited=False)

def BFS_food(start, food_matrix):
        """
        A BFS variant specialized for finding islands of food.
        \n returns a list of positions within the same island as `start`
        """
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                        (1, 1), (-1, 1), (1, -1), (-1, -1)]
        
        def stopcondition(list):
            return False
        
        return breadth_first_search(start, stopcondition=stopcondition,
                                            data_matrix=food_matrix,
                                            neighbor_selection=neighbors,
                                            return_visited=True,
                                            expand_into_true=True)

def get_all_dead_paths(agent, game_state):
    """
    returns a list of all paths resulting in a dead end, which are in turn lists of positions.
    This function is expensive and should only be called once at the beginning of each game.
    """
    dead_paths = []    
    for dead_end in get_dead_ends(agent, game_state):
        dead_list = BFS_walls(agent, dead_end)
        dead_paths.append(dead_list)
    return dead_paths

def get_closest_enemy_distance(agent, game_state, successor):
    """
    returns the distance (int) to closest enemy agent (with noise)
    """
    enemiesList = agent.get_opponents(successor)
    agentDistances =  successor.get_agent_distances()
    enemyDistances = []
    enemyStates = []
    closestEnemyDist = float("+inf")

    ## gathering smallest distance from enemies and also the index of the closest enemy
    for index, x in enumerate(enemiesList):
        enemyDistances.append(agentDistances[x])
        enemyStates.append(game_state.get_agent_state(x))

        if enemyDistances[index] < closestEnemyDist:
            closestEnemyDist = enemyDistances[index]

    return min(enemyDistances)

def get_teammate_index(agent, game_state):
    """
    returns the index of your teammate (int). Should only need to be called only once at the beginning of each game.
    """    
    teammate_idx = None
    for index in agent.get_team(game_state):
        if index != agent.index:
            teammate_idx = index
    return teammate_idx

def get_missing_food(agent, CurrentTeamFood):
    """
    returns tuple (x,y) of the most recently eaten fooddot if one has been eaten in the past 30 actions, returns None otherwise.
    """
    missing_food = []
    prev_observation = agent.get_previous_observation()
    if prev_observation is not None:
        PrevTeamFood = agent.get_food_you_are_defending(prev_observation)
        missing_food = [element for element in PrevTeamFood.as_list() if element not in CurrentTeamFood.as_list()]

    missing_food_updated = False
    if len(missing_food) > 0:
        missing_food_updated = True

    if missing_food_updated:
        agent.eaten_fooddot = missing_food[0]
        agent.clock = 1
    if not missing_food_updated:
        agent.clock += 1
    if agent.clock > 30:
        agent.eaten_fooddot = None
    
    return missing_food[0] if missing_food else None

def getDistFromMiddle(agent, agent_idx, successor):
    """
    returns the smallest distance (int) to the midline for a given agent index and successor.
    """
    dist = float("+inf")
    agent_pos = successor.get_agent_position(agent_idx)
    for pos in agent.midline:
        dist = min(dist,agent.get_maze_distance(pos,agent_pos))
    return dist

def get_neighbor_food(agent, x,y, food_matrix):
    """
    returns a tuple of lists: (foods, non_foods)
    \n foods is a list of positions that contain a food dot
    \n non_foods is a list of positions that do not contain a food dot
    """
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                    (1, 1), (-1, 1), (1, -1), (-1, -1)]
    return agent.get_neighbor_data(x,y,food_matrix,neighbors)

def get_food_islands(agent, food_list, food_matrix):
    """
    returns a list of islands, which are in turn lists of positions of food dots.
    \nIt does so by calling `breadth_first_search_food` for each food dot in `food_list`
    """
    islands = []
    visited = set()
    
    for food in food_list:
        if not food in visited:
            visited.add(food)
            island = BFS_food(food,food_matrix)
            islands.append(island)
    return islands

def get_largest_food_island(agent):
    """
    returns the largest island in agent.food_islands
    """
    return max(agent.food_islands, key=len)

def update_food_islands(agent, curr_pos, food_list, food_matrix):
    """
    Updates food islands only when needed to save computation. There are two cases:
    \nThe position of the agent is the start position (game start or respawn)
    \nThe largest food island has been completely eaten
    """ 
    if curr_pos == agent.start:
        agent.food_islands = get_food_islands(agent,food_list,food_matrix)

    agent.largest_island = get_largest_food_island(agent)

    largest_still_exists = False
    for pos in agent.largest_island:
        if food_matrix[pos[0]][pos[1]]:
            largest_still_exists = True
    
    if not largest_still_exists:
        agent.food_islands = get_food_islands(agent,food_list,food_matrix) ## recalc if largest food island has been eaten

def distance_from_island(agent, island, succ_pos):
    """
    returns the distance (int) to the closest food dot from a given island
    """
    min_distance = float("+inf")
    for pos in island:
        min_distance = min(min_distance, agent.get_maze_distance(succ_pos,pos))
    return min_distance

def check_for_capsule_consumption(agent, capsules_list, time_left):
    """
    Updates agent.most_recent_capsule_consumption if a capsule has been eaten
    """
    missing_capusles = []
    prev_observation = agent.get_previous_observation()
    if prev_observation is not None:
        Prevcaps = agent.get_capsules(prev_observation)
        missing_capusles = [element for element in Prevcaps if element not in capsules_list]

    if missing_capusles:
        agent.most_recent_capsule_consumption = time_left

def validate_position(agent, position, game_state):
    """
    returns a non-wall position, given a position. If the given position does not contain a wall, then it is returned. 
    \nOtherwise it will give the first found neighbor which is not occupied by a wall
    """
    x,y = position

    if not game_state.has_wall(int(x) , int(y)):
        return (int(x),int(y))
    
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                    (1, 1), (-1, 1), (1, -1), (-1, -1)]
    
    for dx, dy in neighbors:
        new_x, new_y = (int(x) + dx, int(y) + dy)
        if not game_state.has_wall(new_x,new_y):
            return (new_x,new_y)

def get_avg_position_from_list(agent, position_list, game_state):
    """
    returns the average position of all positions in `position_list`
    \nIf `position_list` is empty, then it returns the center of the of the teamside.
    """
    if not position_list:
        return (int(agent.x_mid + agent.x_mid/2), int(agent.height/2)) ## mistake here: hardcoded for blue side center...
    curr_x = 0
    curr_y = 0
    for pos in position_list:
        curr_x += pos[0]
        curr_y += pos[1]
    avg_x = curr_x/len(position_list)
    avg_y = curr_y/len(position_list)
    return validate_position(agent, (avg_x,avg_y), game_state)

def get_powerup_deadline(agent, scared_ghosts, turns_left, powerup_deadline):
    """
    returns the estimated remaining turns (int) with scared ghosts.
    """
    if scared_ghosts:
        return scared_ghosts[0].scared_timer
    else:
        return turns_left - powerup_deadline if agent.most_recent_capsule_consumption > 0 else 0

def get_midline(agent, game_state):
    """
    sets agent.midline to the list of all positions where x = agent.walls.width/2
    """
    agent.midline = []
    for y in range(0,agent.height):
        if not game_state.has_wall(agent.x_mid,y):
            agent.midline.append((agent.x_mid,y))

def get_invader_distance(agent, succ_pos, game_state, invaders, features):
    """
    returns the distance to the nearest invader. If there are no detected invaders, then returns agent.eaten_fooddot. If there is no eaten fooddot, then returns 0.
    Always returns an integer.
    """
    if not any_pacman_from_enemies(agent, game_state):
        return 0

    dist = agent.get_maze_distance(succ_pos, agent.eaten_fooddot) if agent.eaten_fooddot else None
    dist = min([agent.get_maze_distance(succ_pos, a.get_position()) for a in invaders]) if invaders else 0
    
    return dist

def check_for_dead_end(agent, succ_pos, succes_agent_state, non_scared_ghosts, features):
    """
    returns a numeric score based on if a successor position is contained within a dead end.
    \n True: -99999999
    \n False: 0
    """
    for dead_path in agent.dead_paths:
        if succ_pos in dead_path or succes_agent_state.is_pacman:
            return -99999999 if non_scared_ghosts and features["ghost_distance"] <= 2*len(dead_path) + 1 else 0
        else:
            return 0

## predicates for decision making
    
def all_pacman_on_team(agent, game_state):
    team_indices = agent.get_team(game_state)
    for index in team_indices:
        if not game_state.get_agent_state(index).is_pacman:
            return False
    return True
    
def any_pacman_from_enemies(agent, game_state):
    for index in agent.enemies:
        if game_state.get_agent_state(index).is_pacman:
            return True
    return False

def closest_to_midline(agent, teammate_idx, successor):
    teammate_home_distance = getDistFromMiddle(agent,teammate_idx, successor)
    my_home_distance = getDistFromMiddle(agent, agent.index, successor)
    if my_home_distance == teammate_home_distance:
        return agent.index == min(agent.get_team(successor))
    return my_home_distance == min(my_home_distance,teammate_home_distance)

def closest_to_pacman(agent, teammate_pos, curr_pos, invader_pos):
    teammate_distance = agent.get_maze_distance(teammate_pos, invader_pos)
    my_distance = agent.get_maze_distance(curr_pos, invader_pos)
    return my_distance == min(my_distance,teammate_distance)

def should_i_defend(agent, game_state, teammate_idx, invader_pos, curr_pos, teammate_position, successor):
    if not game_state.get_agent_state(agent.index).is_pacman and game_state.get_agent_state(teammate_idx).is_pacman:
        ## I am ghost and buddy is pacman -> me
        return True
    elif game_state.get_agent_state(agent.index).is_pacman and not game_state.get_agent_state(teammate_idx).is_pacman:
        ## I am pac and buddy is ghost
        return False
    elif game_state.get_agent_state(agent.index).is_pacman and game_state.get_agent_state(teammate_idx).is_pacman:
        ##both pacman
        return closest_to_pacman(agent, teammate_position, curr_pos, invader_pos) if invader_pos else closest_to_midline(agent, teammate_idx, successor)
    else:
        ## both ghost
        if any_pacman_from_enemies(agent,game_state):
            return closest_to_pacman(agent,teammate_position, curr_pos, invader_pos) if invader_pos else closest_to_midline(agent,teammate_idx, successor)
        else:
            return not closest_to_pacman(agent,teammate_position, curr_pos, invader_pos) if invader_pos else closest_to_midline(agent,teammate_idx, successor)

def get_agent_features(agent, game_state, action):
    features = util.Counter()
    #agent.debug_clear()

    ## variables
    successor = agent.get_successor(game_state, action)
    current_agent_state = game_state.data.agent_states[agent.index]
    succes_agent_state = successor.data.agent_states[agent.index]
    curr_pos = current_agent_state.get_position()
    succ_pos = succes_agent_state.get_position()

    time_left = game_state.data.timeleft
    turns_left = int(time_left/4)
    scared_timer = succes_agent_state.scared_timer
    is_scared = scared_timer > 0

    closest_enemy_distance = get_closest_enemy_distance(agent,game_state,successor)

    enemy_states = [successor.get_agent_state(i) for i in agent.get_opponents(successor)]
    enemy_scared_timers = [enemy.scared_timer for enemy in enemy_states]
    enemy_scared_factor = sum(enemy_scared_timers)/len(enemy_scared_timers)

    invaders = [a for a in enemy_states if a.is_pacman and a.get_position() is not None]
    ghosts = [a for a in enemy_states if not a.is_pacman and a.get_position() is not None]
    scared_ghosts = [enemy for enemy in enemy_states if not enemy.is_pacman and enemy.scared_timer > 0]
    non_scared_ghosts = [a for a in ghosts if a.scared_timer == 0]
    non_scared_distances = [agent.get_maze_distance(succ_pos, a.get_position()) for a in non_scared_ghosts] if non_scared_ghosts else [0]
    teamCapsules = agent.get_capsules_you_are_defending(game_state) ## list[(x,y)] caps on our side
    CurrentTeamFood = agent.get_food_you_are_defending(game_state) ## matrix with true/false
    
    teammate_idx = get_teammate_index(agent,game_state)
    teammate_position = game_state.get_agent_position(teammate_idx)
            
    invader_pos = get_missing_food(agent,CurrentTeamFood)

    previous_positions = get_prev_positions(agent)
    bad_positions = get_bad_positions(previous_positions)

    food_matrix = agent.get_food(successor)
    food_list = food_matrix.as_list()
    min_distance_food = min([agent.get_maze_distance(succ_pos, food) for food in food_list])
    update_food_islands(agent, curr_pos, food_list, food_matrix)

    capsules_list = agent.get_capsules(successor)
    min_distance_cap = min([agent.get_maze_distance(succ_pos, cap) for cap in capsules_list]) if len(capsules_list) > 0 else 0
    
    check_for_capsule_consumption(agent,capsules_list, time_left)
    powerup_deadline = agent.most_recent_capsule_consumption - 40 if agent.most_recent_capsule_consumption > 0 else 100000
    powerup_remaining_time = get_powerup_deadline(agent,scared_ghosts, turns_left, powerup_deadline)
        
    no_dangerghosts = 5 if not non_scared_ghosts else 0
    retreat_threshold = 5 + enemy_scared_factor*0.2 + no_dangerghosts
    
    retreat_mode = 1 if current_agent_state.num_carrying >= retreat_threshold else 0
    double_attack = 1 if all_pacman_on_team(agent, game_state) else 0

    if powerup_remaining_time > 0:
        if powerup_remaining_time <= min_distance_cap:
            features['distance_to_capsule'] = min_distance_cap
        else:
            features['distance_to_capsule'] = min_distance_cap if powerup_remaining_time == 0 else min_distance_cap/powerup_remaining_time
        features["remaining_capsules"] = -len(capsules_list)
    else:
        features["remaining_capsules"] = len(capsules_list)
        features['distance_to_capsule'] = min_distance_cap if powerup_remaining_time == 0 else min_distance_cap/powerup_remaining_time
        

    features['distance_to_food'] = min_distance_food
    features["distance_to_largest_food_island"] = distance_from_island(agent, agent.largest_island, succ_pos)
    features["closest_enemy_dist"] = closest_enemy_distance
    features["remaining_food"] = len(food_list)
    features["return_urgency"] = -getDistFromMiddle(agent, agent.index, successor)*retreat_mode
    features['successor_score'] = agent.get_score(successor)
    features["spread_tendency"] = agent.get_maze_distance(succ_pos,teammate_position)*double_attack
    features['num_invaders'] = len(invaders)
    features["capsule_middle_distance"] = agent.get_maze_distance(get_avg_position_from_list(agent,teamCapsules, game_state),succ_pos) if any_pacman_from_enemies(agent, game_state) and len(teamCapsules) > 0 else 0
    features["center_ownside_distance"] = agent.get_maze_distance(get_avg_position_from_list(agent,food_list, game_state),succ_pos) if agent.active_profile == "defend" else 0
    features["dont_die"] = -999999 if succ_pos == agent.start else 0
    features["anti_stuck"] = 1 if succ_pos in bad_positions else 0
    features["ghost_distance"] = min(non_scared_distances)
    features["invader_distance"] = get_invader_distance(agent,succ_pos,game_state, invaders, features)
    features["barely_evade"] = 1 if features['invader_distance'] == 1 and is_scared else 0
    features["no_dead_end"] = check_for_dead_end(agent,succ_pos,succes_agent_state,non_scared_ghosts,features)

    if action == Directions.STOP:
        features['stop'] = 1 if not bad_positions else 2000

    rev = Directions.REVERSE[game_state.get_agent_state(agent.index).configuration.direction]
    if action == rev: features['reverse'] = 1 if not bad_positions else 10    

    ## determine what profile will be used
    if should_i_defend(agent, game_state, teammate_idx, invader_pos, curr_pos, teammate_position, successor) and any_pacman_from_enemies(agent, game_state):
        agent.active_profile = "defend"
    else:
        agent.active_profile = "attack"

    return features

class ApproximateFridgeAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.prev_weights = {'distance_to_food': -1.2750175913589804e+122,
                            'distance_to_capsule': -1.4378496172883322e+122,
                            'remaining_capsules': -5.425495352743764e+120,
                            'closest_enemy_dist': -4.611688429895748e+121, 
                            'remaining_food': -8.137719306164774e+121, 
                            'return_urgency': -7.463680033535487e+31, 
                            'successor_score': 3.7973578685836735e+121, 
                            'spread_tendency': 26202841201473.695, 
                            'num_invaders': -2.712573100149988e+120, 
                            'capsule_middle_distance': -7.052131332296071e+121, 
                            'center_ownside_distance': -9.49522786507858e+121, 
                            'dont_die': 2.712895094168435e+125, 
                            'anti_stuck': -2.7125730970237865e+120, 
                            'ghost_distance': -1.3195836269405794e+113, 
                            'invader_distance': -9.766345559279176e+121, 
                            'barely_evade': 0.0, 
                            'stop': -5.425146194047573e+123, 
                            'reverse': -2.7129222275061062e+125, 
                            'no_dead_end': 0,
                            'distance_to_largest_food_island' : 0}
        
        self.discount = 0.5 ## falloff
        self.alpha = 0.01 ## learning rate

    ## Overrides from (Reflex)CaptureAgent
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

    def register_initial_state(self, game_state):
        """
        Some important values are initialized at the very first game_state.
        `get_maze_distance` is called, alongside some other functions that need to be called only once,
        such as `get_midline`, `get_all_dead_paths` and `get_opponents`.
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.distancer.get_maze_distances()
        self.active_profile = "attack"
        self.weights = self.prev_weights

        self.most_recent_capsule_consumption = 0
        self.clock = 0
        self.eaten_fooddot = None
        self.enemies = self.get_opponents(game_state)

        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.x_mid = int(self.width/2) if not self.red else int(self.width/2) - 1
        get_midline(self, game_state)

        self.dead_paths = get_all_dead_paths(self,game_state)

    def get_features(self, game_state, action):
        return get_agent_features(self,game_state,action)

    def choose_action(self, game_state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = game_state.get_legal_actions(self.index)

        max_q = self.compute_value_from_q_values(game_state)

        best_actions = []

        # make a list of all of the actions that, combined with the state, have the maximum q-value
        # then choose randomly from that list
        for action in actions:
            if self.get_q_value(game_state, action) == max_q:
                best_actions.append(action)
       
        if not best_actions:
            return game_state.get_legal_actions(self.index)[0]

        return random.choice(best_actions)
    
    ## methods from QLearningAgent
    def compute_value_from_q_values(self, game_state):
        """
          Returns max_action Q(state, action)
          where the max is over legal actions.  Notere that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        value = float ("-inf") # values can be negative, so initialize at -inf
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return 0.0 # default value
       
        for action in actions:
            value = max(value, (self.get_q_value(game_state, action)))
        return value

    def get_weights(self):
        return self.weights

    def get_q_value(self, state, action):
        """
          Should return Q(state, action) = w * feature_vector
          where * is the dot_product operator
        """
        # Christmas present from Arno
        weights = self.get_weights()
        features = self.get_features(state, action)


        # Q(s, a) = w_1 f_1(s, a) + w_2 f_2(s, a) + ... + w_n f_n(s, a)
        dot_product = 0
        for feature_name, feature_value in features.items():
            dot_product += weights[feature_name] * feature_value
        return dot_product

    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        weights = self.get_weights()
        features = self.get_features(state, action)


        # compute max q-value of all state-acion pairs of a state
        def max_q_value(state):
            actions = state.get_legal_actions(self.index)
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
        prev_prev_observation = self.observation_history[-3]
        prev_prev_action = self.get_action(prev_prev_observation)
        prev_prev_score = self.get_score(prev_prev_observation)
        prev_observation = self.get_previous_observation()
        prev_action = self.get_action(prev_observation)
        prev_score = self.get_score(prev_observation)
        delta_reward = prev_score - prev_prev_score
        self.update(prev_prev_observation,prev_prev_action,prev_observation,0.01)
        print(self.weights)

        self.prev_weights = self.weights

        # call the super-class final method
        CaptureAgent.final(self, state)

class SmartFridgeAgent(ReflexCaptureAgent):
        
    def register_initial_state(self, game_state):
        """
        Some important values are initialized at the very first game_state.
        `get_maze_distance` is called, alongside some other functions that need to be called only once,
        such as `get_midline`, `get_all_dead_paths` and `get_opponents`.
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.distancer.get_maze_distances()

        self.active_profile = "attack"

        self.most_recent_capsule_consumption = 0
        self.clock = 0
        self.eaten_fooddot = None
        self.enemies = self.get_opponents(game_state)

        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.x_mid = int(self.width/2) if not self.red else int(self.width/2) - 1
        get_midline(self, game_state)

        self.dead_paths = get_all_dead_paths(self,game_state)

    def get_features(self, game_state, action):
        return get_agent_features(self, game_state, action)
    
    def get_weights(self, game_state, action):
        """
        returns a set of weights (dictionary) based on `self.active_profile`
        """
        
        attack_profile = {
            "successor_score": 200,
            "remaining_food": -100,
            "remaining_capsules": -500,
            "distance_to_food": -5,
            "distance_to_largest_food_island": -9,
            "distance_to_capsule": -20,
            "ghost_distance": 10,
            "return_urgency": 1000,
            "anti_stuck": -1000,
            "spread_tendency": 2,
            "num_invaders": -1000,
            "invader_distance": -5,
            "barely_evade": -5000,
            "stop": -100,
            "reverse": -1,
            "dont_die": 1,
            "no_dead_end": 1}
        
        defend_profile = {
            "num_invaders": -1000,
            "invader_distance": -30,
            "capsule_middle_distance": -10,
            "center_ownside_distance": -5,
            "closest_enemy_dist": -2,
            "anti_stuck": -1000,
            "barely_evade": -5000, 
            "spread_tendency": 10,
            "stop": -100,
            "reverse": -1,
            "dont_die": 1,
            "successor_score": 10}
        
        chosen_profile = defend_profile if self.active_profile == "defend" else attack_profile
        return chosen_profile
