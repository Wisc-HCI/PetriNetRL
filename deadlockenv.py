import gymnasium
from gymnasium import spaces
import numpy as np
import sys

def is_goal(marking, goal_state):
    places = np.where(goal_state == 1)[0]
    for i in places:
        if marking[i][0] < 1:
            return False
    return True
	
class DeadlockEnv(gymnasium.Env):
    """Environment for learning to avoid deadlock situations within the petrinet"""

    def __init__(self, json_obj):
        super(DeadlockEnv, self).__init__()

        self.num_places = len(json_obj["places"])
        self.num_transitions = len(json_obj["transitions"])

        self.place_names = [json_obj["places"][i]["name"] for i in json_obj["places"]]
        self.transition_names = [json_obj["transitions"][i]["name"] for i in json_obj["transitions"]]

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
        # For deadlock, we want to avoid "rest" actions - transitions that input and output are the same
        self.base_mask = [True for _ in range(self.num_transitions)]

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

        # iterate over transitions and see which ones are "rest" actions
        for i, transition in enumerate(json_obj["transitions"]):
            is_rest_action = True
            for place in json_obj["transitions"][transition]["input"]:
                if not is_rest_action:
                    continue
                if place not in json_obj["transitions"][transition]["output"] or
                    json_obj["transitions"][transition]["input"][place]["value"] != json_obj["transitions"][transition]["output"][place]["value"]:
                    is_rest_action = False

            if is_rest_action:
                # Add transition to rest list so we can mark it invalid in mask
                self.base_mask[i] = False


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
        self.previous_state = self.marking.copy()
        self.marking = self.marking + np.dot(self.C, a)

        # Determine reward
        tmp_rwd = self.get_reward(self.marking)
        done = tmp_rwd < 0 or is_goal(self.marking, self.goal_state)

        return self.marking, tmp_rwd, done, False, {}

    def reset(self, seed=0, options={}):
        self.done = False
        self.marking = self.initial_marking.copy()
        return self.marking, {}  # reward, done, info can't be included

    def get_reward(self, newMarking):
        valid_actions = [True for _ in range(self.num_transitions)]
        for i in range(self.num_transitions):
            column = np.empty((self.num_places, 1))
            for j in range(self.num_places):
                if newMarking[j][0] + self.iC[j][i] < 0:
                    valid_actions[i] = False
        
        if any(valid_actions):
            return 0.0

        # No valid actions, so this is a bad state to be in
        return -99.0

    def valid_action_mask(self):
        valid_actions = self.base_mask.copy()
        for (j, i) in self.non_zeros:
            if not valid_actions[i]:
                continue
            # Mark any transitions that would cause an invalid state (value < 0) as invalid
            if self.marking[j][0] + self.iC[j][i] < 0:
                valid_actions[i] = False
        return valid_actions