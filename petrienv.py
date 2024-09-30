import sys
import gymnasium
from gymnasium import spaces
import numpy as np
from constants import *
import random

def is_goal(marking, goal_state):
    places = np.where(goal_state == 1)[0]
    for i in places:
        if marking[i][0] < 1:
            return False
    return True

class PetriEnv(gymnasium.Env):
    """Environment for learning the collaborative task"""

    def __init__(self, json_obj):
        super(PetriEnv, self).__init__()

        self.json_obj = json_obj
        self.num_places = len(json_obj["places"])
        self.num_transitions = len(json_obj["transitions"])
        self.available_workers = []

        self.all_agents = []
        self.busy_workers = []
        self.agent_transitions = {}
        self.current_time = 0

        self.place_names = [json_obj["places"][i]["name"] for i in json_obj["places"]]
        self.transition_names = []
        self.transition_times = []
        self.transition_ids = []
        for i in json_obj["transitions"]:
            self.transition_ids.append(i)
            self.transition_names.append(json_obj["transitions"][i]["name"])
            self.transition_times.append(json_obj["transitions"][i]["time"])

        # Build cost array - ordered as [OneTime, Extrapolated]
        # TODO: will need to check type and alpha, balancing between ERGO and ECON
        self.transition_costs = [[0, 0] for _ in range(self.num_transitions)]
        self.used_one_time_cost = [False for _ in range(self.num_transitions)]
        for i, transition in enumerate(json_obj["transitions"]):
            one_time_cost = 0
            extrapolated_cost = 0
            for c in json_obj["transitions"][transition]["cost"]:
                if c["frequency"] == ONE_TIME_KEY:
                    one_time_cost -= c["value"]
                else:
                    extrapolated_cost -= c["value"]
            self.transition_costs[i] = [one_time_cost, extrapolated_cost]
     
            # Build network of all transitions relevant to specific agents
            for data in json_obj["transitions"][transition]["metaData"]:
                if data["type"] == "agent":
                    if data["value"] not in self.all_agents:
                        self.all_agents.append(data["value"])

                    try:
                        self.agent_transitions[data["value"]].append(i)
                    except:
                        self.agent_transitions[data["value"]] = [i]


        self.initial_marking = np.empty((self.num_places, 1))
        self.goal_state = np.empty((self.num_places, 1))
        for i, place in enumerate(json_obj["places"]):
            if json_obj["places"][place]["tokens"] == "infinite":
                self.initial_marking[i][0] = sys.maxsize
            else:
                try:
                    self.initial_marking[i][0] = json_obj["initialMarking"][place]
                except:
                    self.initial_marking[i][0] = 0

            self.goal_state[i][0] = 1 if json_obj["places"][place]["tokens"] == "sink" and "🗑️" not in json_obj["places"][place]["name"] else 0
        
        self.C = np.empty((self.num_places, self.num_transitions))
        # Input incedence
        self.iC = np.empty((self.num_places, self.num_transitions))
        # Output incedence
        self.oC = np.empty((self.num_places, self.num_transitions))

        # Matrix is sparse, so we can leverage that for the action masking
        self.non_zeros = []

        self.marking = self.initial_marking.copy()

        for row, place in enumerate(json_obj["places"]):
            for col, key in enumerate(json_obj["transitions"]):
                transition = json_obj["transitions"][key]
                delta = 0
                deltaI = 0
                deltaO = 0

                if (place in list(transition["input"])):
                    delta -= transition["input"][place]["value"]
                    deltaI -= transition["input"][place]["value"]

                if (place in list(transition["output"])):
                    delta += transition["output"][place]["value"]
                    deltaO += transition["output"][place]["value"]

                self.C[row][col] = delta
                if deltaI < 0:
                    self.non_zeros.append((row, col))
                self.iC[row][col] = deltaI
                self.oC[row][col] = deltaO

        # 1 action each timestep
        self.action_space = spaces.Discrete(self.num_transitions)

        # Could do a multiple discrete
        # Each transitions can either activate or not
        # You need to figure out whether or not it's valid though
        # self.action_space = spaces.MultiDiscrete([2 for _ in self.num_transitions])
        
        # Number of tokens in each place (marking) - vector of places by 1
        self.observation_space = spaces.Box(low=-255, high=255,
                                            shape=(self.num_places, 1,), dtype=np.float64)

    # State reward function
    def reward_value(self, action, previous_state, new_state, goal_state, current_time):
        # Reward is given by the cost of executing the transition
        reward = self.transition_costs[action][EXTRAPOLATED_INDEX]
        if not self.used_one_time_cost[action]:
            reward += self.transition_costs[action][ONE_TIME_INDEX]
            self.used_one_time_cost[action] = True
        # Add incentive to avoid bad/infeasible states
        if np.any(new_state < 0.0):
            reward += -100000.0
        # Add incentive for the goal state
        elif is_goal(new_state, goal_state):
            reward += 10000.0
        # cost to explore
        # elif reward == 0:
        #     reward -= 0.1 * current_time
        reward -= 0.1
        return reward

    def step(self, action):
        a = np.asarray([[0 if action != j else 1] for j in range(self.num_transitions)])
        self.previous_state = self.marking.copy()
        self.marking = self.marking + np.dot(self.iC, a)

        transition = self.json_obj["transitions"][self.transition_ids[action]]
        for data in transition["metaData"]:
            if data["type"] == "agentAgnostic":
                # TODO: agentAgnostic..... <------
                # figure out who is free, select one, add them to busy worker
                # really only matters if time is > 0
                available_workers = []

                for agent in self.all_agents:
                    available = True
                    for (worker, _time, _action) in self.busy_workers:
                        if not available:
                            continue
                        if worker == agent:
                            available = False
                    if available:
                        available_workers.append(agent)

                # need a better selection method
                selected_worker = random.choice(available_workers)
                if transition["time"] > 0:
                    self.busy_workers.append((selected_worker, self.current_time + transition["time"], a.copy()))
                else:
                    self.busy_workers.append((selected_worker, self.current_time + transition["time"], a.copy()))
            elif data["type"] == "agent":
                self.busy_workers.append((data["value"], self.current_time + transition["time"], a.copy()))

        
        # determine whether to move time forward or not
        if len(self.busy_workers) == len(self.all_agents):
            new_time = min(list(map(lambda pair: pair[1], self.busy_workers)))
            new_busy_workers = []
            for (worker, time, action_vec) in self.busy_workers:
                if time <= new_time:
                    self.marking = self.marking + np.dot(self.oC, action_vec)
                else:
                    new_busy_workers.append((worker, time, action_vec))
            self.busy_workers = new_busy_workers
            # self.busy_workers = list(filter(lambda pair: pair[1] > new_time, self.busy_workers))
            self.current_time = new_time

        # at least one worker should now be free since we moved time forward


        tmp_rwd = self.reward_value(action, self.previous_state, self.marking, self.goal_state, self.current_time)

        return self.marking, tmp_rwd, is_goal(self.marking, self.goal_state), False, {}

    def reset(self, seed=0, options={}):
        self.done = False
        self.marking = self.initial_marking.copy()
        return self.marking, {}  # reward, done, info can't be included


    def valid_action_mask(self):
        valid_actions = [True for _ in range(self.num_transitions)]

        # If worker is busy, they can't perform any new actions, so mark those actions as false
        for (worker_id, _time, _action) in self.busy_workers:
            for transition_index in self.agent_transitions[worker_id]:
                valid_actions[transition_index] = False

        for (j, i) in self.non_zeros:
            if not valid_actions[i]:
                continue
            # Mark any transitions that would cause an invalid state (value < 0) as invalid
            if self.marking[j][0] + self.iC[j][i] < 0:
                valid_actions[i] = False
            # elif "Rest" in self.transition_names[i]:
            #     valid_actions[i] = False
        return valid_actions