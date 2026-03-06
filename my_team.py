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

class SmartFridgeAgent(ReflexCaptureAgent):
        
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.distancer.get_maze_distances()

        self.starting_food = self.get_food_you_are_defending(game_state)
        self.starting_food_amount = len(self.starting_food.as_list())
        self.starting_capsules = self.get_capsules(game_state)

        def get_neighbor_walls(x,y):
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            walls = []
            non_walls = []
            for dx, dy in neighbors:
                neighbor = game_state.data.layout.walls[x+dx][y+dy]
                if neighbor:
                    walls.append((x+dx, y+dy))
                else:
                    non_walls.append((x+dx, y+dy))
            return walls, non_walls
            
        def get_dead_ends():
                """
                    returns a list of ((x,y), empty_neighbor) of all dead end cells
                """
                walls = game_state.get_walls()
                wall_list = walls.as_list()
                neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                dead_ends = []
                for x in range(walls.width):
                    for y in range(walls.height):
                        if not game_state.data.layout.walls[x][y]:
                            wall_neighbors, empty_neighbors = get_neighbor_walls(x,y)
                            if len(wall_neighbors) == 3:
                                ## this is a dead end cell
                                dead_ends.append(((x,y), empty_neighbors[0]))
                                self.debug_draw((x,y),color=(.4,.6,.2))
                return dead_ends

        def breadth_first_search(start):
            agenda = util.Queue()
            init_cell = start
            ## de agenda is een stack van nodes: eerste element is de state, tweede element is het pad tot nu toe
            agenda.push([init_cell,[]])
            Visited = set([init_cell])

            while True:
                if agenda.is_empty():
                    return []

                current_state = agenda.pop()
                current_path = current_state[1]
                current_cell = current_state[0]

                walls, non_walls = get_neighbor_walls(current_cell[0],current_cell[1])
                
                if len(walls) < 2:
                    #self.debug_draw(prev_cell,color=(0.9,0.2,0.2))
                    return current_path

                for next_cell in non_walls:
                    if next_cell not in Visited:
                        Visited.add(next_cell)
                        self.debug_draw(current_cell,color=(0.5,0.8,0.3))
                        agenda.push([next_cell, current_path + [current_cell]])

        def get_all_dead_paths():
            dead_paths = []
            for dead_end, start in get_dead_ends():
                dead_list = breadth_first_search(start)
                dead_paths.append(dead_list)
            return dead_paths

        print(get_all_dead_paths())
            

    def get_features(self, game_state, action):
        features = util.Counter()
        self.debug_clear()

        ## general information
        previous_positions = []
        for observ in self.observation_history[-12: ]:
            position = observ.get_agent_position(self.index)
            previous_positions.append(position)

        uniquePositions = CountList(previous_positions).keys()
        uniqueCount = CountList(previous_positions).values()
        bad_positions = []
        #print(uniquePositions.mapping.get())

        for pos in uniquePositions.mapping:
            count = uniquePositions.mapping.get(pos)
            if count >= 6:
                bad_positions.append(pos)
            

        #print(uniquePositions,"\n", uniqueCount,"\n", previous_positions,"\n")
            

        successor = self.get_successor(game_state, action)
        present_agent_state = game_state.data.agent_states[self.index]
        succes_agent_state = successor.data.agent_states[self.index]

        scared_timer = succes_agent_state.scared_timer
        is_scared = scared_timer > 0

        agentDistances =  successor.get_agent_distances()
        curr_pos = present_agent_state.get_position()
        succ_pos = succes_agent_state.get_position()

        walls = game_state.get_walls()
        width = walls.width
        height = walls.height

        x_mid = int(width/2) if not self.red else int(width/2) - 1

        midline = []
        for y in range(0,height):
            if not game_state.has_wall(x_mid,y):
                midline.append((x_mid,y))

        def getDistFromMiddle(agent_idx):
            dist = float("+inf")
            agent_pos = successor.get_agent_position(agent_idx)
            for pos in midline:
                dist = min(dist,self.get_maze_distance(pos,agent_pos))
            return dist

        ## info about enemies
        enemiesList = self.get_opponents(successor)
        enemyDistances = []
        enemyStates = []
        closestEnemyDist = float("+inf")

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

        enemy_scared_timers = [enemy.scared_timer for enemy in enemies]

        enemy_scared_factor = sum(enemy_scared_timers)/len(enemy_scared_timers)

        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]


        ## gathering smallest distance from enemies and also the index of the closest enemy
        for index, x in enumerate(enemiesList):
            enemyDistances.append(agentDistances[x])
            enemyStates.append(game_state.get_agent_state(x))

            if enemyDistances[index] < closestEnemyDist:
                closestEnemyDist = enemyDistances[index]

        closestEnemyDist = min(enemyDistances)


        ## info about our side
        teamCapsules = self.get_capsules_you_are_defending(game_state) ## list[(x,y)] caps on our side
        CurrentTeamFood = self.get_food_you_are_defending(game_state) ## matrix with true/false
        teammate_idx = None
        for index in self.get_team(game_state):
            if index != self.index:
                teammate_idx = index

        ## computed heuristics
        food_list = self.get_food(successor).as_list()
        min_distance_food = min([self.get_maze_distance(succ_pos, food) for food in food_list])

        capsules_list = self.get_capsules(successor)
        min_distance_cap = min([self.get_maze_distance(succ_pos, cap) for cap in capsules_list]) if len(capsules_list) > 0 else 0

        def get_food_islands(food_positions):
            food_positions = food_positions.as_list()
            islands = []
            visited = set()
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                         (1, 1), (-1, 1), (1, -1), (-1, -1)]
            
            def get_neighbors(pos):
                res = []
                for dx, dy in neighbors:
                    new_pos = (pos[0]+dx, pos[1]+dy)
                    res.append(new_pos)
                return res
            
            for pos in food_positions:
                if not pos in visited:
                    curr_island = set(pos)
                    queue = []

                    while queue:
                        curr = queue.pop()
                        if curr in visited:
                            continue
                        visited.add(curr)
                        curr_island.add(curr)
                        queue.append(get_neighbors(curr))

                    islands.append(curr_island)
            return islands


        #for index, island in enumerate(get_food_islands(self.get_food_you_are_defending(game_state))):
        #    increment = 1/len(get_food_islands(self.get_food_you_are_defending(game_state)))
        #    self.debug_draw(island,color=(increment*index,.5,increment*index))
                


        def all_pacman_on_team():
            team_indices = self.get_team(game_state)
            for index in team_indices:
                if not game_state.get_agent_state(index).is_pacman:
                    return False
            return True
        
        def any_pacman_from_enemies():
            for index in enemiesList:
                if game_state.get_agent_state(index).is_pacman:
                    return True
            return False
        
        def closest_to_midline():
            teammate_home_distance = getDistFromMiddle(teammate_idx)
            my_home_distance = getDistFromMiddle(self.index)
            if my_home_distance == teammate_home_distance:
                return self.index == min(self.get_team(game_state))
            return my_home_distance == min(my_home_distance,teammate_home_distance)

        def closest_pacman(agent_idx):
            closest = float("+inf")
            for index, dist in enumerate(game_state.get_agent_distances()):
                if game_state.get_agent_state(index).is_pacman and not index in self.get_team(game_state): ## agent is an enemy and is a pacman
                    closest = min(closest,dist)
            return closest
        
        def closest_to_pacman():
            return 
        
        def get_buddy_distance():
            teammate_pos = successor.get_agent_position(teammate_idx)
            dist = self.get_maze_distance(succ_pos,teammate_pos)
            return dist
                
        def should_i_defend():
            if not game_state.get_agent_state(self.index).is_pacman and game_state.get_agent_state(teammate_idx).is_pacman:
                ## I am ghost and buddy -> me
                return True
            elif game_state.get_agent_state(self.index).is_pacman and not game_state.get_agent_state(teammate_idx).is_pacman:
                ## I am pac and buddy is ghost
                return False
            elif game_state.get_agent_state(self.index).is_pacman and game_state.get_agent_state(teammate_idx).is_pacman:
                ##both pacman
                return closest_to_midline()
            else:
                ## both ghost
                if any_pacman_from_enemies():
                    return not closest_to_midline()
                else:
                    return closest_to_midline()

        def get_capsule_middle_point():
            curr_x = 0
            curr_y = 0
            for (x,y) in teamCapsules:
                curr_x += x
                curr_y += y
            avg_x = curr_x / len(teamCapsules)
            avg_y = curr_y / len(teamCapsules)
            #self.debug_draw((avg_x,avg_y),color=(0.8,0.2,0.8))
            if game_state.has_wall(int(avg_x) , int(avg_y)):
                if not game_state.has_wall(int(avg_x) + 1 , int(avg_y)):
                    return (int(avg_x) + 1 , int(avg_y))
                elif not game_state.has_wall(int(avg_x) + 1 , int(avg_y) + 1):
                    return (int(avg_x) + 1 , int(avg_y) + 1)
                elif not game_state.has_wall(int(avg_x) , int(avg_y) +1 ):
                    return (int(avg_x) , int(avg_y) +1 )
                elif not game_state.has_wall(int(avg_x) -1 , int(avg_y) + 1):
                    return (int(avg_x) - 1 , int(avg_y + 1))
                elif not game_state.has_wall(int(avg_x) - 1 , int(avg_y)):
                    return (int(avg_x) - 1 , int(avg_y))
                elif not game_state.has_wall(int(avg_x) - 1 , int(avg_y) - 1):
                    return (int(avg_x) - 1 , int(avg_y) - 1)
                elif not game_state.has_wall(int(avg_x) , int(avg_y) - 1):
                    return (int(avg_x) , int(avg_y) - 1)
                elif not game_state.has_wall(int(avg_x) + 1 , int(avg_y) - 1):
                    return (int(avg_x) + 1 , int(avg_y) - 1)
                
            
            return (int(avg_x) , int(avg_y))

        def get_your_half_center():
            curr_x = 0
            curr_y = 0
            for pos in CurrentTeamFood.as_list():
                curr_x += pos[0]
                curr_y += pos[1]
            avg_x = curr_x/len(CurrentTeamFood.as_list())
            avg_y = curr_y/len(CurrentTeamFood.as_list())
            #self.debug_draw((avg_x,avg_y),color=(0.2,0.1,0.8))
            if game_state.has_wall(int(avg_x) , int(avg_y)):
                if not game_state.has_wall(int(avg_x) + 1 , int(avg_y)):
                    return (int(avg_x) + 1 , int(avg_y))
                elif not game_state.has_wall(int(avg_x) + 1 , int(avg_y) + 1):
                    return (int(avg_x) + 1 , int(avg_y) + 1)
                elif not game_state.has_wall(int(avg_x) , int(avg_y) +1 ):
                    return (int(avg_x) , int(avg_y) +1 )
                elif not game_state.has_wall(int(avg_x) -1 , int(avg_y) + 1):
                    return (int(avg_x) - 1 , int(avg_y + 1))
                elif not game_state.has_wall(int(avg_x) - 1 , int(avg_y)):
                    return (int(avg_x) - 1 , int(avg_y))
                elif not game_state.has_wall(int(avg_x) - 1 , int(avg_y) - 1):
                    return (int(avg_x) - 1 , int(avg_y) - 1)
                elif not game_state.has_wall(int(avg_x) , int(avg_y) - 1):
                    return (int(avg_x) , int(avg_y) - 1)
                elif not game_state.has_wall(int(avg_x) + 1 , int(avg_y) - 1):
                    return (int(avg_x) + 1 , int(avg_y) - 1)
            return (int(avg_x) , int(avg_y))
            
        def avg_of_two_pos(pos1, pos2):
            total_x = pos1[0] + pos2[0]
            total_y = pos1[1] + pos2[1]
            return (total_x*0.5, total_y*0.5)

        retreat_threshold = 5 + enemy_scared_factor*0.2

        retreat_mode = 9999999 if succes_agent_state.num_carrying >= retreat_threshold else 1

        double_attack = 1 if all_pacman_on_team() else 0

        features['distance_to_food'] = min_distance_food
        features['distance_to_capsule'] = min_distance_cap
        features["remaining_capsules"] = len(capsules_list)
        features["closest_enemy_dist"] = closestEnemyDist
        features["remaining_food"] = len(food_list)
        features["return_urgency"] = -getDistFromMiddle(self.index)*retreat_mode
        features['successor_score'] = self.get_score(successor)
        features["spread_tendency"] = get_buddy_distance()*double_attack
        features['num_invaders'] = len(invaders)
        features["capsule_middle_distance"] = self.get_maze_distance(get_capsule_middle_point(),succ_pos) if any_pacman_from_enemies() and len(teamCapsules) > 0 else 0
        features["center_ownside_distance"] = self.get_maze_distance(get_your_half_center(),succ_pos)
        max_tweak_dist = 0

        for pos in bad_positions:
            max_tweak_dist = max(self.get_maze_distance(succ_pos,pos),max_tweak_dist)

        features["anti-tweak"] = max_tweak_dist

        if any_pacman_from_enemies():
            dists = None
            if len(invaders) > 0:
                dists = [self.get_maze_distance(succ_pos, a.get_position()) for a in invaders]
            else:
                dists = enemyDistances
            features['invader_distance'] = min(dists)

        features["barely_evade"] = 1 if features['invader_distance'] == 1 and is_scared else 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1 if not bad_positions else 1000    

        ## determine what profile will be used
        if should_i_defend() and any_pacman_from_enemies():
            self.active_profile = "defend"
        else:
            self.active_profile = "attack"

        if self.active_profile == "defend":
            self.debug_draw(curr_pos,color=(0.8,0.3,0.3))
        elif self.active_profile == "attack":
            self.debug_draw(curr_pos,color=(0.3,0.8,0.3))
        else:
            print("no profile")
            
        if len(teamCapsules) > 0:
            self.debug_draw(avg_of_two_pos(get_capsule_middle_point(),get_your_half_center()),color=(1,1,1))

        return features
        
    def get_weights(self, game_state, action):
        attack_profile = {"distance_to_food": -5, "distance_to_capsule": -10, "remaining_capsules": -1000 , "closest_enemy_dist": 10,
                           "remaining_food": -1000, "return_urgency": 2, "successor_score": 1,
                             "spread_tendency": 4, "num_invaders": -1000, "invader_distance": -11,
                              "stop": -100, "reverse": -1, "capsule_middle_distance": 0,
                              "barely_evade": -99999, "anti-tweak": -50}
        
        defend_profile = {"num_invaders": -1000, "invader_distance": -10, "stop": -100, "reverse": -2,
                            "capsule_middle_distance": 0, "closest_enemy_dist": -2, "center_ownside_distance": -5,
                            "barely_evade": -99999, "spread_tendency": 5, "anti-tweak": -50}
        
        
        chosen_profile = defend_profile if self.active_profile == "defend" else attack_profile
        return chosen_profile


