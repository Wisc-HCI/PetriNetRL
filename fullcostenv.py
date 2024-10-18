import sys
import gymnasium
from gymnasium import spaces
import numpy as np
from constants import *
from utils import *
import random
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

class FullCostEnv(gymnasium.Env):
    """Environment for learning the collaborative task"""

    def __init__(self, json_obj, weights):
        super(FullCostEnv, self).__init__()

        self.json_obj = json_obj

        # Store cost weighting
        self.weights = [weights[ERGO_KEY], weights[MONETARY_KEY]]
        
        # Get the number of places and transitions in the petri net
        self.num_places = len(json_obj["places"])
        self.num_transitions = len(json_obj["transitions"])

        # Trackers for whether workers are busy/free
        self.busy_workers = []

        # Tracker for what agents exist in the network
        self.all_agents = []

        # Index into all transition specific to an agent
        self.agent_transitions = {}

        # Time tracker
        self.current_time = 0

        # Get the names of the places in the petri net
        self.place_names = [json_obj["places"][i]["name"] for i in json_obj["places"]]

        
        # Get the names, ids, and times of the transitions in the petri net
        self.transition_names = []
        self.transition_times = []
        self.transition_ids = []
        for i in json_obj["transitions"]:
            self.transition_ids.append(i)
            self.transition_names.append(json_obj["transitions"][i]["name"])
            self.transition_times.append(json_obj["transitions"][i]["time"])

        # Build cost array - ordered as [OneTime, Extrapolated]
        self.transition_costs = [[0, 0] for _ in range(self.num_transitions)]

        # Tracker for whether an action has been used for the first time or not
        self.used_one_time_cost = [False for _ in range(self.num_transitions)]

        # Iterate over each transition to determine cost for it to be used
        for i, transition in enumerate(json_obj["transitions"]):
            one_time_cost = 0
            extrapolated_cost = 0
            
            # Look over the cost set and add up the one time and extraploted costs (to each one's respective position in the array)
            for c in json_obj["transitions"][transition]["cost"]:
                # Determine alpha weighting for each category
                multiplier = 1
                if c["category"] == ERGO_KEY:
                    multiplier = self.weights[0]
                else:
                    multiplier = self.weights[1]

                # Apply weighting to costs
                if c["frequency"] == ONE_TIME_KEY:
                    one_time_cost += multiplier * c["value"]
                else:
                    extrapolated_cost += multiplier * c["value"]

            self.transition_costs[i] = [one_time_cost, extrapolated_cost]
     
            # Inspect transition metadata to see which agent it is assigned to
            for data in json_obj["transitions"][transition]["metaData"]:
                if data["type"] == "agent":
                    # If agent data is found, ensure that the agent is in the all_agents tracker
                    if data["value"] not in self.all_agents:
                        self.all_agents.append(data["value"])

                    # Add the transition to the agent's possible action space
                    try:
                        self.agent_transitions[data["value"]].append(i)
                    except:
                        self.agent_transitions[data["value"]] = [i]

        # Determine the initial marking and goal state(s) of the petri net
        self.initial_marking = np.empty((self.num_places, 1))
        self.goal_state = np.empty((self.num_places, 1))
        for i, place in enumerate(json_obj["places"]):
            # Places of infinite token have max value
            if json_obj["places"][place]["tokens"] == "infinite":
                self.initial_marking[i][0] = sys.maxsize
            # Otherwise is an infinite sink or has a predefined size
            else:
                # Try to lookup initial marking count for place
                try:
                    self.initial_marking[i][0] = json_obj["initialMarking"][place]
                # If no initial marking is found, assign 0
                except:
                    self.initial_marking[i][0] = 0

            # Mark goal state for states that are sinks, but not discards
            self.goal_state[i][0] = 1 if json_obj["places"][place]["tokens"] == "sink" and "🗑️" not in json_obj["places"][place]["name"] else 0
        
        # Overall incedence matrix
        self.C = np.empty((self.num_places, self.num_transitions))

        # Input incedence
        self.iC = np.empty((self.num_places, self.num_transitions))

        # Output incedence
        self.oC = np.empty((self.num_places, self.num_transitions))

        # Matrix is sparse, so we can leverage that for the action masking
        self.non_zeros = []

        # Set the starting marking to the be the initial one
        self.marking = self.initial_marking.copy()

        # Build the various incidence matricies
        for row, place in enumerate(json_obj["places"]):
            for col, key in enumerate(json_obj["transitions"]):
                transition = json_obj["transitions"][key]
                delta = 0
                deltaI = 0
                deltaO = 0
                
                # For each input into a transition, this "takes away" value or consumes tokens from a place
                # So we subtract that value from the matrix's cell value
                if (place in list(transition["input"])):
                    delta -= transition["input"][place]["value"]
                    deltaI -= transition["input"][place]["value"]

                # For each output from a transition, this places a token into a place
                # So we add that value to the matrix's cell value
                if (place in list(transition["output"])):
                    delta += transition["output"][place]["value"]
                    deltaO += transition["output"][place]["value"]

                # The incidence matrix cell is just the output - inputs
                self.C[row][col] = delta

                # If the number of tokens consumed by the transition is non-zero, track it
                if deltaI < 0:
                    self.non_zeros.append((row, col))
                    
                # Update the input incidence matrix cell value
                self.iC[row][col] = deltaI

                # Update the output incidence matrix cell value
                self.oC[row][col] = deltaO

        # Setup the action space such that it is 1 action per call of the step function
        self.action_space = spaces.Discrete(self.num_transitions)

        # TODO: Could do a multiple discrete
        # Each transitions can either activate or not
        # You need to figure out whether or not it's valid though
        # self.action_space = spaces.MultiDiscrete([2 for _ in self.num_transitions])
        
        # Setup observational space to be the number of tokens in each place (marking) - i.e. vector of places by 1
        self.observation_space = spaces.Box(low=-255, high=255,
                                            shape=(self.num_places, 1,), dtype=np.float32)

    # State reward function
    def reward_value(self, action, previous_state, new_state, goal_state, current_time):
        """Reward function for the full-cost environment. All costs come from the transitions"""
        reward = 0

        # Reward is given by the cost of executing the transition
        reward -= self.transition_costs[action][EXTRAPOLATED_INDEX]

        # If transition has a 1 time cost (such as purchasing) and it hasn't been used before, use it
        if not self.used_one_time_cost[action]:
            # Incur transition cost
            reward -= self.transition_costs[action][ONE_TIME_INDEX]

            # Update that we have incurred the cost
            self.used_one_time_cost[action] = True
        
        # Check if any invalid states occur (place has negative token count)
        if np.any(new_state < 0.0):
            reward += INVALID_STATE_REWARD
        # Check if the goal conidition are met
        elif IS_GOAL(new_state, goal_state):
            reward += GOAL_FOUND_REWARD
        
        # Incur a very small cost to explore
        reward += STEP_REWARD

        return reward

    def step(self, action):
        # Build action array of 0s except for the selected action
        a = np.asarray([[0 if action != j else 1] for j in range(self.num_transitions)])

        # Store the previous marking before updating
        self.previous_state = self.marking.copy()
        
        # Update the marking by the dot product of the input incidence matrix and the action vector
        self.marking = self.marking + np.dot(self.iC, a)

        # Get the transition of the selected action
        transition = self.json_obj["transitions"][self.transition_ids[action]]

        # Determine who to assign the work to
        for data in transition["metaData"]:
            # If there is an agentAgnostic metadata, anyone can perform the action
            if data["type"] == "agentAgnostic":
                # TODO: agentAgnostic..... <------
                # figure out who is free, select one, add them to busy worker
                # really only matters if time is > 0

                # Tracking list of available workers
                available_workers = []

                # Iterate over all agents
                for agent in self.all_agents:
                    available = True

                    # Check if the agent is already in the busy worker list
                    # If so, they aren't available
                    for (worker, _time, _action) in self.busy_workers:
                        if not available:
                            continue
                        if worker == agent:
                            available = False

                    # Add available agents to the list
                    if available:
                        available_workers.append(agent)

                # Randomly select agents from the available pool
                # TODO: need a better selection method
                selected_worker = random.choice(available_workers)

                # Update the busy workers list
                self.busy_workers.append((selected_worker, self.current_time + transition["time"], a.copy()))
            # If the transition has the agent metadata, it is assigned to that specific agent
            # No need to check if they are in busy_workers, since the mask function should invalidate any actions related to busy workers
            elif data["type"] == "agent":
                self.busy_workers.append((data["value"], self.current_time + transition["time"], a.copy()))

        
        # Determine whether to move time forward or not
        # If all agents are allocated, we need to advance time
        if len(self.busy_workers) == len(self.all_agents):
            # Find the smallest time interval to advance by
            new_time = min(list(map(lambda pair: pair[1], self.busy_workers)))

            # Update the busy workers list to account for this time change (freeing up at least 1 worker)
            new_busy_workers = []
            for (worker, time, action_vec) in self.busy_workers:
                # If worker is freed up, we can update the marking to account for the completion of the task
                if time <= new_time:
                    self.marking = self.marking + np.dot(self.oC, action_vec)
                # Otherwise, worker is still busy
                else:
                    new_busy_workers.append((worker, time, action_vec))
        
            # Set busy worker list
            self.busy_workers = new_busy_workers

            # self.busy_workers = list(filter(lambda pair: pair[1] > new_time, self.busy_workers))
            self.current_time = new_time

        # At least one worker should now be free (for the next step call) since we moved time forward

        # Determine reward
        tmp_rwd = self.reward_value(action, self.previous_state, self.marking, self.goal_state, self.current_time)

        return self.marking, tmp_rwd, IS_GOAL(self.marking, self.goal_state), False, {}

    def reset(self, seed=0, options={}):
        """Reset the environment"""

        self.done = False

        # Reset the marking to the default state of the petri net
        self.marking = self.initial_marking.copy()

        return self.marking, {}  # reward, done, info can't be included

    def valid_action_mask(self):
        """Determine all possible valid actions at the current state"""

        # Assume all actions are valid
        valid_actions = [True for _ in range(self.num_transitions)]

        # If worker is busy, they can't perform any new actions, so mark any actions related to that worker as false
        for (worker_id, _time, _action) in self.busy_workers:
            for transition_index in self.agent_transitions[worker_id]:
                valid_actions[transition_index] = False

        # Iterate over non-zero transitions
        for (j, i) in self.non_zeros:
            # Ignore already false transitions
            if not valid_actions[i]:
                continue

            # Mark any transitions that would cause an invalid state (value < 0) as invalid
            if self.marking[j][0] + self.iC[j][i] < 0:
                valid_actions[i] = False

        return valid_actions