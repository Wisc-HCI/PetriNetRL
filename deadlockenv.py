import gymnasium
from gymnasium import spaces
import numpy as np
import sys
from constants import *
from utils import *
	
class DeadlockEnv(gymnasium.Env):
    """Environment for learning to avoid deadlock situations within the petrinet"""

    def __init__(self, json_obj):
        super(DeadlockEnv, self).__init__()

        # Get the number of places and transitions in the petri net
        self.num_places = len(json_obj["places"])
        self.num_transitions = len(json_obj["transitions"])

        # Get the names of the places and transitions in the petri net
        self.place_names = [json_obj["places"][i]["name"] for i in json_obj["places"]]
        self.transition_names = [json_obj["transitions"][i]["name"] for i in json_obj["transitions"]]

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

        # For deadlock, we want to avoid "rest" actions - transitions that the input and output are the same
        self.base_mask = [True for _ in range(self.num_transitions)]

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

        # iterate over transitions and see which ones are "rest" actions
        # we remove these from the base mask, since you can "rest" infinitely
        # TODO: can also do this with spawning, but we need spawning to make progress - how do we account for this?
        for i, transition in enumerate(json_obj["transitions"]):
            shortcut = False
            for data in json_obj["transitions"][transition]["metaData"]:
                if shortcut:
                    continue
                if data["type"] == "rest":
                    # Add transition to rest list so we can mark it invalid in mask
                    self.base_mask[i] = False
                    shortcut = True

        # Set the starting marking to the be the initial one
        self.marking = self.initial_marking.copy()

        # Setup the action space such that it is 1 action per call of the step function
        self.action_space = spaces.Discrete(self.num_transitions)

        # TODO: Could do a multiple discrete
        # Each transitions can either activate or not
        # Need to figure out whether or not it's valid though
        # self.action_space = spaces.MultiDiscrete([2 for _ in self.num_transitions])
        
        # Setup observational space to be the number of tokens in each place (marking) - i.e. vector of places by 1
        self.observation_space = spaces.Box(low=-255, high=255,
                                            shape=(self.num_places, 1,), dtype=np.float64)

    def step(self, action):
        # Build action array of 0s except for the selected action
        a = np.asarray([[0 if action != j else 1] for j in range(self.num_transitions)])
        
        # Update the marking by the dot product of the incidence matrix and the action vector
        self.marking = self.marking + np.dot(self.C, a)

        # Determine reward
        tmp_rwd = self.get_reward(self.marking.copy())

        # Check if goal state is reached
        goal_reached = IS_GOAL(self.marking, self.goal_state)

        # high reward for goal
        # if goal_reached:
        #     tmp_rwd += GOAL_FOUND_REWARD
        
        # Mark done if reward is negative or if the goal state is reached
        done = tmp_rwd < 0 or goal_reached

        return self.marking.copy(), tmp_rwd, done, False, {}

    def reset(self, seed=0, options={}):
        """Reset the environment"""

        self.done = False

        # Reset the marking to the default state of the petri net
        self.marking = self.initial_marking.copy()

        return self.marking, {}  # reward, done, info can't be included

    def get_reward(self, newMarking):
        """Reward function for the deadlock environment"""

        # Tracker for whether all agents in the petri net have been discarded
        allAgentsDiscarded = True

        # Iterate over all places in the petrinet where agents are discarded (1 for each agent)
        for i in self.discard_places:
            # Check if the agent discard location is empty (if so, agent hasn't been discarded)
            if allAgentsDiscarded and newMarking[i][0] == 0:
                allAgentsDiscarded = False
        
        # If all agents are discard, this is a deadlock so assign heavy negative reward
        if allAgentsDiscarded:
            return DEADLOCK_OCCURS_REWARD

        # Iterate over transitions (action space) and check if any actions are available from the current state
        # for i in range(self.num_transitions):
        #     for j in range(self.num_places):
        #         if newMarking[j][0] + self.iC[j][i] >= 0:
        #             return 0.0

        # Iterate over valid transitions and check if any are available from the current state
        # If so, return 0.0
        for (j, i) in self.non_zeros:
            if self.marking[j][0] + self.iC[j][i] >= 0:
                return 0.0

        # No valid actions, so this is a bad state to be in
        return DEADLOCK_OCCURS_REWARD

    def valid_action_mask(self):
        """Determine all possible valid actions at the current state"""

        # Copy the base mask
        valid_actions = self.base_mask.copy()

        # Iterate over non-zero transitions
        for (j, i) in self.non_zeros:
            # Skip non-valid transitions
            if not valid_actions[i]:
                continue

            # Mark any transitions that would cause an invalid state (value < 0) as invalid
            if self.marking[j][0] + self.iC[j][i] < 0:
                valid_actions[i] = False

        return valid_actions

if __name__ == "__main__":
    deadlock_env_id = 'wiscHCI/DeadlockEnv-v0'

    gymnasium.envs.registration.register(
        id=deadlock_env_id,
        entry_point=DeadlockEnv,
        max_episode_steps=MAX_DEADLOCK_ITERATIONS,
        reward_threshold=500 # TODO: No clue what value to use here...
    )