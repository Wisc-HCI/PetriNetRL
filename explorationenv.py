import gymnasium
from gymnasium import spaces
import numpy as np
import sys
from constants import *
from utils import *
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)
	
class ExplorationEnv(gymnasium.Env):
    """Environment for exploring the petrinet. Actions in this environment are free, penalizing for deadlocks and invalid states, and rewarded for finding the goal state and making progress towards it"""

    def __init__(self, json_obj, json_task):
        super(ExplorationEnv, self).__init__()

        # Get the number of places and transitions in the petri net
        self.num_places = len(json_obj["places"])
        self.num_transitions = len(json_obj["transitions"])

        # Get the names of the places in the petri net
        self.place_names = [json_obj["places"][i]["name"] for i in json_obj["places"]]
        
        # Get the names and ids of the transitions in the petri net
        self.transition_ids = []
        self.transition_names = []
        for i in json_obj["transitions"]:
            self.transition_ids.append(i)
            self.transition_names.append(json_obj["transitions"][i]["name"])

        # Determine all discard places (should be 1 for each agent)
        self.discard_places = []
        for i in json_obj["places"]:
            if "🗑️" in json_obj["places"][i]["name"]:
                self.discard_places.append(self.place_names.index(json_obj["places"][i]["name"]))

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
        
        # Incedence matrix
        self.C = np.empty((self.num_places, self.num_transitions))

        # Input incedence
        self.iC = np.empty((self.num_places, self.num_transitions))

        # The matrix is sparse, so we can leverage that for the action masking
        self.non_zeros = []

        # Create a base mask of acceptable actions
        self.base_mask = [True for _ in range(self.num_transitions)]


        self.tasks = json_task
        # for idx, task_id in enumerate(json_task):
        #     self.tasks.append(task_id)

        # Tracking whether "first time reward" can be applied for this transition
        # I am basing this off of "making progress" in the collaboration - i.e. they produce an intermediate part / use a target
        self.first_time_reward_for_transition = [False for _ in range(self.num_transitions)]
        self.task_transitions = [-1 for _ in range(self.num_transitions)]
        self.setup_transition = [False for _ in range(self.num_transitions)]

        # Build input and overall incidence matricies
        for row, place in enumerate(json_obj["places"]):
            for col, key in enumerate(json_obj["transitions"]):
                transition = json_obj["transitions"][key]
                delta = 0
                deltaI = 0

                # For each input into a transition, this "takes away" value or consumes tokens from a place
                # So we subtract that value from the matrix's cell value
                if (place in list(transition["input"])):
                    delta -= transition["input"][place]["value"]
                    deltaI -= transition["input"][place]["value"]

                # For each output from a transition, this places a token into a place
                # So we add that value to the matrix's cell value
                if (place in list(transition["output"])):
                    delta += transition["output"][place]["value"]

                # The incidence matrix cell is just the output - inputs
                self.C[row][col] = delta

                # If the number of tokens consumed by the transition is non-zero, track it
                if deltaI < 0:
                    self.non_zeros.append((row, col))

                # Update the input incidence matrix cell value
                self.iC[row][col] = deltaI

        # iterate over transitions and mark all actions that are related to setup or task progression as available for reward
        for i, place in enumerate(json_obj["transitions"]):
            is_progressable_action = False
            is_setup_step = False
            for data in json_obj["transitions"][place]["metaData"]:
                if data["type"] == "setup":
                    is_setup_step = True
                if data["type"] == "task":
                    is_progressable_action = True

                    # Offset by 1 since this is multiplied in the reward function
                    self.task_transitions[i] = self.tasks[data["value"]]["order"]

            # Mark whether the transition is a setup step or if it is a task-based transition
            if is_setup_step:
                self.setup_transition[i] = True
            elif is_progressable_action:
                self.first_time_reward_for_transition[i] = True


        # print(self.task_transitions)
        # Set the starting marking to the be the initial one
        self.marking = self.initial_marking.copy()

        # Setup the action space such that it is 1 action per call of the step function
        self.action_space = spaces.Discrete(self.num_transitions)

        # TODO: Could do a multiple discrete
        # Each transitions can either activate or not
        # You need to figure out whether or not it's valid though
        # self.action_space = spaces.MultiDiscrete([2 for _ in self.num_transitions])
        
        # Setup observational space to be the number of tokens in each place (marking) - i.e. vector of places by 1
        self.observation_space = spaces.Box(low=-255, high=255,
                                            shape=(self.num_places, 1,), dtype=np.float32)

    def step(self, action):
        # Build action array of 0s except for the selected action
        a = np.asarray([[0 if action != j else 1] for j in range(self.num_transitions)])
        
        # Update the marking by the dot product of the incidence matrix and the action vector
        self.marking = self.marking + np.dot(self.C, a)

        # Determine reward
        tmp_rwd = self.get_reward(self.marking.copy(), action)

        # Check if goal state is reached
        goal_reached = IS_GOAL(self.marking, self.goal_state)

        # Mark done if reward is negative or if the goal state is reached
        done = goal_reached

        return self.marking.copy(), tmp_rwd, done, False, {}

    def reset(self, seed=0, options={}):
        """Reset the environment"""

        self.done = False

        # Reset the marking to the default state of the petri net
        self.marking = self.initial_marking.copy()

        return self.marking, {}  # reward, done, info can't be included

    def get_reward(self, newMarking, chosenAction):
        """Reward function for the exploration environment"""

        reward = 0

        # Tracker for whether all agents in the petri net have been discarded
        allAgentsDiscarded = True
        
        # Iterate over all places in the petrinet where agents are discarded (1 for each agent)
        for i in self.discard_places:
            # Check if the agent discard location is empty (if so, agent hasn't been discarded)
            if allAgentsDiscarded and newMarking[i][0] == 0:
                allAgentsDiscarded = False
        
        # If all agents are discard, this is a deadlock so assign heavy negative reward
        if allAgentsDiscarded:
            reward += DEADLOCK_OCCURS_REWARD

        # Iterate over transitions (action space) and check if any actions are available from the current state
        # If so, return 0.0
        valid_actions = False
        i = 0
        while i < self.num_transitions:
            j = 0
            while j < self.num_places:
                if newMarking[j][0] + self.iC[j][i] >= 0:
                    valid_actions = True
                    i = self.num_transitions
                    j = self.num_places
                j += 1
            i += 1

        # No valid actions, so this is a bad state to be in
        if not valid_actions:
            reward += DEADLOCK_OCCURS_REWARD


        goal_reached = IS_GOAL(newMarking, self.goal_state)

        # Check if new state is invalid or goal
        if np.any(newMarking < 0.0):
            reward += INVALID_STATE_REWARD
        elif goal_reached:
            reward += GOAL_FOUND_REWARD

        # If not in a deadlock, invalid, or goal state, check if this is a setup step
        # If so, reward
        if reward >= 0 and self.setup_transition[chosenAction]:
            # Small incentive to progress to goal
            reward += 2.0 * FIRST_TIME_ACTION_REWARD

        if reward >= 0 and self.first_time_reward_for_transition[chosenAction]:
            # self.first_time_reward_for_transition[chosenAction] = False
            # Small incentive to progress to goal
            if self.task_transitions[chosenAction] > -1:
                reward += self.task_transitions[chosenAction] * FIRST_TIME_ACTION_REWARD
            else:
                reward += FIRST_TIME_ACTION_REWARD

        # Add cost for exploration
        reward += STEP_REWARD

        return reward
    

        

        # # If transition has a 1 time cost (such as purchasing) and it hasn't been used before, use it
        # if not self.used_one_time_cost[action]:
        #     # Incur transition cost
        #     reward -= self.transition_costs[action][ONE_TIME_INDEX]

        #     # Update that we have incurred the cost
        #     self.used_one_time_cost[action] = True

    def valid_action_mask(self):
        """Determine all possible valid actions at the current state"""

        # Copy the base mask
        valid_actions = self.base_mask.copy()

        # Iterate over non-zero transitions
        for (j, i) in self.non_zeros:
            # Ignore already false transitions
            if not valid_actions[i]:
                continue

            # Mark any transitions that would cause an invalid state (value < 0) as invalid
            if self.marking[j][0] + self.iC[j][i] < 0:
                valid_actions[i] = False

        return valid_actions

if __name__ == "__main__":
    exploration_env_id = 'wiscHCI/ExplorationEnv-v0'

    gymnasium.envs.registration.register(
        id=exploration_env_id,
        entry_point=ExplorationEnv,
        max_episode_steps=MAX_EXPLORATION_ITERATIONS,
        reward_threshold=500 # TODO: No clue what value to use here...
    )