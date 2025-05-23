import sys
import gymnasium
from gymnasium import spaces
import numpy as np
from constants import *
from utils import *
import random
import math
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

class FullCostEnv(gymnasium.Env):
    """Environment for learning the collaborative task"""

    def __init__(self, json_obj, weights, tasks, targets, agent_obj):
        super(FullCostEnv, self).__init__()

        self.json_obj = json_obj
        self.tasks = tasks
        self.targets = targets

        # Store cost weighting
        self.weights = [weights[ERGO_KEY], weights[MONETARY_KEY]]
        
        # Get the number of places and transitions in the petri net
        self.num_places = len(json_obj["places"])
        self.num_transitions = len(json_obj["transitions"])

        # Trackers for whether workers are busy/free
        self.busy_workers = []

        # Tracker for what agents exist in the network
        self.all_agents = []
        self.all_agents_reset = []
        self.discarded_agents = []
        self.agent_obj = agent_obj

        # Tracker for agent exertion rate
        self.agent_exertion = []

        # Index into all transition specific to an agent
        self.agent_transitions = {}

        # Time tracker
        self.current_time = 0

        self.step_tracker = 0

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

        # Tracking whether "first time reward" can be applied for this transition
        # I am basing this off of "making progress" in the collaboration - i.e. they produce an intermediate part / use a target
        self.first_time_reward_for_transition = [False for _ in range(self.num_transitions)]
        self.task_transitions = [-1 for _ in range(self.num_transitions)]
        self.setup_transition = [False for _ in range(self.num_transitions)]

        # Tracking # of agent transitions + allocations
        self.min_setup_transitions = []
        self.min_number_of_setup_actions = 0

        self.base_mask = [True for _ in range(self.num_transitions)]

        self.rest_action_indecies = []
        self.toggled_base_mask = False

        self.discard_actions = []

        # Iterate over each transition to determine cost for it to be used
        for i, transition in enumerate(json_obj["transitions"]):
            one_time_cost = 0
            extrapolated_cost = 0

            # TODO: do something if rest action? (i.e. recoup costs if costs exist)
            
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
     
            is_progressable_action = False
            is_setup_step = False
            # Inspect transition metadata to see which agent it is assigned to
            for data in json_obj["transitions"][transition]["metaData"]:
                if data["type"] == "agent":
                    # If agent data is found, ensure that the agent is in the all_agents tracker
                    if data["value"] not in self.all_agents:
                        self.all_agents.append(data["value"])
                        self.all_agents_reset.append(data["value"])
                        self.agent_exertion.append([0, 0])

                    # Add the transition to the agent's possible action space
                    try:
                        self.agent_transitions[data["value"]].append(i)
                    except:
                        self.agent_transitions[data["value"]] = [i]
                elif data["type"] == "rest":
                    # Add transition to rest list so we can mark it invalid in mask
                    # self.base_mask[i] = False
                    self.rest_action_indecies.append(i)
                elif data["type"] == "agentDiscard":
                    self.discard_actions.append(i)
                elif data["type"] == "setup":
                    is_setup_step = True
                elif data["type"] == "task":
                    is_progressable_action = True

                    # Offset by 1 since this is multiplied in the reward function
                    self.task_transitions[i] = self.tasks[data["value"]]["order"] + 1

            # Mark whether the transition is a setup step or if it is a task-based transition
            if is_setup_step:
                self.setup_transition[i] = True
                self.min_setup_transitions.append(i)
            elif is_progressable_action:
                self.first_time_reward_for_transition[i] = True

        num_precusors = 0
        num_reusable = 0
        for target in self.targets:
            if self.targets[target]["type"] == "reusable":
                num_reusable += 1
            elif self.targets[target]["type"] == "precursor":
                num_precusors += 1

        # Update variable tracking the min number of steps needed to complete the setup phase
        self.min_number_of_setup_actions = len(self.all_agents) + len(tasks) + num_precusors + num_reusable

        # Determine all discard places (should be 1 for each agent)
        self.discard_places = []
        for i in json_obj["places"]:
            if "🗑️" in json_obj["places"][i]["name"]:
                self.discard_places.append(self.place_names.index(json_obj["places"][i]["name"]))

        # Find min/max of the extrapolated and one time costs
        maxValue = -sys.maxsize - 1
        minValue = sys.maxsize
        maxValue2 = -sys.maxsize - 1
        minValue2 = sys.maxsize
        for i in range(len(self.transition_costs)):
            if self.transition_costs[i][0] >= 0:
                maxValue = max(maxValue, self.transition_costs[i][0])
                minValue = min(minValue, self.transition_costs[i][0])

            if self.transition_costs[i][1] >= 0:
                maxValue2 = max(maxValue2, self.transition_costs[i][1])
                minValue2 = min(minValue2, self.transition_costs[i][1])

        # Normalize (0-1) based on the found min/maxs
        for i in range(len(self.transition_costs)):
            if self.transition_costs[i][0] > 0:
                self.transition_costs[i][0] = (self.transition_costs[i][0]-minValue) / (maxValue - minValue)
            if self.transition_costs[i][1] > 0:
                self.transition_costs[i][1] = (self.transition_costs[i][1]-minValue2) / (maxValue2 - minValue2)

        # Determine the initial marking and goal state(s) of the petri net
        self.initial_marking = np.empty((self.num_places, 1))
        self.goal_state = np.empty((self.num_places, 1))

        # Add additional length for exertion rates
        self.observation = np.empty((self.num_places + len(self.all_agents), 1))

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
                                            shape=(self.num_places+len(self.agent_exertion), 1,), dtype=np.float32)

    # State reward function
    def reward_value(self, action, previous_state, new_state, goal_state, current_time, selected_workers):
        """Reward function for the full-cost environment. All costs come from the transitions"""
        reward = 0
        
        # If all agents are discard, this is a deadlock so assign heavy negative reward
        if ALL_AGENTS_DISCARDED(new_state, self.discard_places) or len(self.all_agents) == 0:
            reward += DEADLOCK_OCCURS_REWARD

        # If transition has a 1 time cost (such as purchasing) and it hasn't been used before, use it
        if not self.used_one_time_cost[action]:
            # Incur transition cost
            reward -= self.transition_costs[action][ONE_TIME_INDEX]

            # Update that we have incurred the cost
            self.used_one_time_cost[action] = True
        
        # Check if any invalid states occur (place has negative token count)
        if IS_INVALID_STATE(new_state):
            reward += INVALID_STATE_REWARD
        # Check if the goal conidition are met
        elif IS_GOAL(new_state, goal_state):
            reward += GOAL_FOUND_REWARD

        
        # If not in a deadlock, invalid, or goal state, check if this is a setup step
        # If so, reward
        if self.setup_transition[action]:
            # Small incentive to progress to goal
            reward += FIRST_TIME_ACTION_REWARD

        for s_worker in selected_workers:
            worker_index = self.all_agents.index(s_worker)
            if action in self.rest_action_indecies and self.agent_exertion[worker_index][TASK_TIME] > 0:
                reward += math.exp(self.agent_exertion[worker_index][EXERTION_TIME] / self.agent_exertion[worker_index][TASK_TIME]) - 1.5

        if self.first_time_reward_for_transition[action]:
            # self.first_time_reward_for_transition[chosenAction] = False
            # Small incentive to progress to goal
            if self.task_transitions[action] > -1:
                reward += self.task_transitions[action] * FIRST_TIME_ACTION_REWARD
            else:
                reward += FIRST_TIME_ACTION_REWARD

            # Reward is given by the cost of executing the transition
        #     reward += self.transition_costs[action][EXTRAPOLATED_INDEX]
        # else:
        reward -= self.transition_costs[action][EXTRAPOLATED_INDEX]

        # Incur a very small cost to explore
        reward += STEP_REWARD

        return reward

    def step(self, action):
        # Build action array of 0s except for the selected action
        a = np.asarray([[0 if action != j else 1] for j in range(self.num_transitions)])

        self.step_tracker += 1

        # Store the previous marking before updating
        self.previous_state = self.marking.copy()
        
        # Update the marking by the dot product of the input incidence matrix and the action vector
        self.marking = self.marking + np.dot(self.iC, a)

        # Get the transition of the selected action
        transition = self.json_obj["transitions"][self.transition_ids[action]]

        selected_worker = None
        all_selected_workers = []
        
        task_start_time = self.current_time
        task_end_time = self.current_time

        
        if action in self.discard_actions:
            agent_id = None

            for data in transition["metaData"]:
                if data["type"] == "agentDiscard":
                    agent_id = data["value"]

            if agent_id is not None:
                removal_index = self.all_agents.index(agent_id)
                self.discarded_agents.append(agent_id)
                del self.all_agents[removal_index]
            else:
                print("WE HAVE A PROBLEM!!!!!")

        # Determine who to assign the work to
        if action not in self.discard_actions and len(self.all_agents) > 0:
            addedAgent = False
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
                    all_selected_workers.append(selected_worker)

                    # Update worker's rest/exertion time
                    if action not in self.rest_action_indecies and selected_worker in self.all_agents:
                        self.agent_exertion[self.all_agents.index(selected_worker)][EXERTION_TIME] += transition["time"]
                    self.agent_exertion[self.all_agents.index(selected_worker)][TASK_TIME] += transition["time"]
                    task_end_time = self.current_time + transition["time"]

                    # Update the busy workers list
                    if not addedAgent:
                        addedAgent = False
                        # Only add 1 copy of the action per agent collaboration
                        self.busy_workers.append((selected_worker, self.current_time + transition["time"], a.copy()))
                    else:
                        self.busy_workers.append((selected_worker, self.current_time + transition["time"], None))
                # If the transition has the agent metadata, it is assigned to that specific agent
                # No need to check if they are in busy_workers, since the mask function should invalidate any actions related to busy workers
                elif data["type"] == "agent":
                    selected_worker = data["value"]
                    task_time = transition["time"]

                    # Shortcut the "rest" action time to better align with agents
                    if action in self.rest_action_indecies and len(self.busy_workers) > 0:
                        next_agent_completion_time = min(list(map(lambda pair: pair[1], self.busy_workers))) - self.current_time
                        if next_agent_completion_time > 0:
                            task_time = min(next_agent_completion_time, task_time)

                    # Update worker's rest/exertion time
                    if action not in self.rest_action_indecies and selected_worker in self.all_agents:
                        self.agent_exertion[self.all_agents.index(selected_worker)][EXERTION_TIME] += transition["time"]

                    if action in self.rest_action_indecies:
                        self.agent_exertion[self.all_agents.index(selected_worker)][TASK_TIME] += task_time
                    else:
                        self.agent_exertion[self.all_agents.index(selected_worker)][TASK_TIME] += transition["time"]

                    task_end_time = self.current_time + task_time
                    
                    all_selected_workers.append(selected_worker)

                    if not addedAgent:
                        addedAgent = True
                        # Only add 1 copy of the action per agent collaboration
                        self.busy_workers.append((data["value"], self.current_time + task_time, a.copy()))
                    else:
                        self.busy_workers.append((data["value"], self.current_time + task_time, None))

        # Determine whether to move time forward or not
        # If all agents are allocated, we need to advance time
        if (self.step_tracker <= self.min_number_of_setup_actions) or (len(self.busy_workers) >= len(self.all_agents) and len(self.all_agents) > 0):
                # Find the smallest time interval to advance by
                temp_list = list([pair[1] for pair in self.busy_workers])
                if len(temp_list) == 0:
                    temp_list.append(0)
                new_time = min(temp_list)

                # Update the busy workers list to account for this time change (freeing up at least 1 worker)
                new_busy_workers = []
                for (worker, time, action_vec) in self.busy_workers:
                    # If worker is freed up, we can update the marking to account for the completion of the task
                    if time <= new_time:
                        # if none, ignore we've already factored it in with the other collaborating agent
                        if action_vec is not None:
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
        tmp_rwd = self.reward_value(action, self.previous_state, self.marking, self.goal_state, self.current_time, all_selected_workers)

        done_flag = IS_GOAL(self.marking, self.goal_state) or IS_INVALID_STATE(self.marking) or ALL_AGENTS_DISCARDED(self.marking, self.discard_places) or len(self.all_agents) == 0

        self.observation[0:self.num_places] = self.marking
        for idx, exertion in enumerate(self.agent_exertion):
            self.observation[self.num_places+idx] = 0 if exertion[TASK_TIME] == 0 else exertion[EXERTION_TIME] / exertion[TASK_TIME]

        return self.observation, tmp_rwd, done_flag, False, {"startTime": task_start_time, "endTime": task_end_time, "busyAgents": [self.agent_obj[s_worker]["name"] for s_worker in all_selected_workers]}

    def reset(self, seed=0, options={}):
        """Reset the environment"""

        self.done = False

        self.step_tracker = 0

        # Reset the marking to the default state of the petri net
        self.marking = self.initial_marking.copy()


        # Reset 1 time costs
        self.used_one_time_cost = [False for _ in range(self.num_transitions)]

        # Trackers for whether workers are busy/free
        self.busy_workers = []

        self.discarded_agents = []

        self.all_agents = self.all_agents_reset.copy()

        # Tracker for agent exertion rate
        for i in range(len(self.agent_exertion)):
            self.agent_exertion[i] = [0, 0]

            
        self.observation[0:self.num_places] = self.marking
        for idx, _ in enumerate(self.agent_exertion):
            self.observation[self.num_places+idx] = 0

        # Time tracker
        self.current_time = 0

        return self.observation, {"time": self.current_time, "busyAgents": self.busy_workers}  # reward, done, info can't be included

    def valid_action_mask(self):
        """Determine all possible valid actions at the current state"""

        # if self.step_tracker >= (FULL_COST_TIMESTEPS*MAX_FULL_COST_ITERATIONS/2) and not self.toggled_base_mask:
        #     self.toggled_base_mask = True
        #     for i in self.rest_action_indecies:
        #         self.base_mask[i] = True
        
        if self.step_tracker <= self.min_number_of_setup_actions:
            # Assume all actions are invalid
            valid_actions = [False for _ in range(self.num_transitions)]

            # Only allow setup actions initially
            for id in self.min_setup_transitions:
                valid_actions[id] = True
        else:
            # Assume all actions are valid
            valid_actions = self.base_mask.copy()

            # If worker is busy, they can't perform any new actions, so mark any actions related to that worker as false
            for (worker_id, _time, _action) in self.busy_workers:
                for transition_index in self.agent_transitions[worker_id]:
                    valid_actions[transition_index] = False

            for agent_id in self.discarded_agents:
                for transition_index in self.agent_transitions[agent_id]:
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