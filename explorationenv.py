import gymnasium
from gymnasium import spaces
import numpy as np
import sys
from constants import *

def is_goal(marking, goal_state):
    places = np.where(goal_state == 1)[0]
    for i in places:
        if marking[i][0] < 1:
            return False
    return True
	
class ExplorationEnv(gymnasium.Env):
    """Environment for learning to avoid deadlock situations within the petrinet"""

    def __init__(self, json_obj):
        super(ExplorationEnv, self).__init__()

        self.num_places = len(json_obj["places"])
        self.num_transitions = len(json_obj["transitions"])

        self.place_names = [json_obj["places"][i]["name"] for i in json_obj["places"]]
        self.transition_names = [json_obj["transitions"][i]["name"] for i in json_obj["transitions"]]
        
        self.transition_ids = []
        for i in json_obj["transitions"]:
            self.transition_ids.append(i)

        self.discard_places = []
        for i in json_obj["places"]:
            if "🗑️" in json_obj["places"][i]["name"]:
                self.discard_places.append(self.place_names.index(json_obj["places"][i]["name"]))

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
        # Matrix is sparse, so we can leverage that for the action masking
        self.non_zeros = []
        # For deadlock, we want to avoid "rest" actions - transitions that input and output are the same
        self.base_mask = [True for _ in range(self.num_transitions)]
        # Tracking whether "first time reward" can be applied for this transition
        # I am basing this off of "making progress" in the collaboration - i.e. they produce an intermediate part / use a target
        self.first_time_reward_for_transition = [False for _ in range(self.num_transitions)]

        for row, place in enumerate(json_obj["places"]):
            for col, key in enumerate(json_obj["transitions"]):
                transition = json_obj["transitions"][key]
                delta = 0
                deltaI = 0

                if (place in list(transition["input"])):
                    delta -= transition["input"][place]["value"]
                    deltaI -= transition["input"][place]["value"]

                if (place in list(transition["output"])):
                    delta += transition["output"][place]["value"]

                self.C[row][col] = delta
                if deltaI < 0:
                    self.non_zeros.append((row, col))
                self.iC[row][col] = deltaI

        # iterate over transitions and find all target actions that aren't related to setup or spawning
        for i, place in enumerate(json_obj["transitions"]):
            has_target = False
            more_targets = False
            for data in json_obj["transitions"][place]["metaData"]:
                if "target" in data["type"] and data["type"] != "target":
                    more_targets = True
                if data["type"] == "target":
                    has_target = True
            if (not more_targets) and has_target:
                self.first_time_reward_for_transition[i] = True

        self.marking = self.initial_marking.copy()

        # 1 action each timestep
        self.action_space = spaces.Discrete(self.num_transitions)

        # Could do a multiple discrete
        # Each transitions can either activate or not
        # You need to figure out whether or not it's valid though
        # self.action_space = spaces.MultiDiscrete([2 for _ in self.num_transitions])
        
        # Number of tokens in each place (marking) - vector of places by 1
        self.observation_space = spaces.Box(low=-255, high=255,
                                            shape=(self.num_places, 1,), dtype=np.float64)

    def step(self, action):
        # Build action array
        a = np.asarray([[0 if action != j else 1] for j in range(self.num_transitions)])
        
        # Update the marking
        self.marking = self.marking + np.dot(self.C, a)

        # Determine reward
        tmp_rwd = self.get_reward(self.marking.copy())
        goal_reached = is_goal(self.marking, self.goal_state)

        if tmp_rwd >= 0 and self.first_time_reward_for_transition[action]:
            self.first_time_reward_for_transition[action] = False
            # Small incentive to progress to goal
            tmp_rwd += 100

        # high reward for goal
        if goal_reached:
            tmp_rwd += 100000
        done = tmp_rwd < 0 or goal_reached

        return self.marking.copy(), tmp_rwd, done, False, {}

    def reset(self, seed=0, options={}):
        self.done = False
        self.marking = self.initial_marking.copy()
        return self.marking, {}  # reward, done, info can't be included

    def get_reward(self, newMarking):
        allAgentsDiscarded = True
        for i in self.discard_places:
            # Check if the agent discard location is empty (if so, agent hasn't been discarded)
            if allAgentsDiscarded and newMarking[i][0] == 0:
                allAgentsDiscarded = False
        
        if allAgentsDiscarded:
            return -99999.0

        for i in range(self.num_transitions):
            for j in range(self.num_places):
                if newMarking[j][0] + self.iC[j][i] >= 0:
                    return 0.0

        # No valid actions, so this is a bad state to be in
        return -99999.0

    def valid_action_mask(self):
        valid_actions = self.base_mask.copy()
        for (j, i) in self.non_zeros:
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