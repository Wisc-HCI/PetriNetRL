import sys
import gymnasium
from gymnasium import spaces
import numpy as np

def is_goal(marking, goal_state):
    places = np.where(goal_state == 1)[0]
    for i in places:
        if marking[i][0] < 1:
            return False
    return True

def reward_value(previous_state, new_state, goal_state):
    if np.array_equal(new_state, previous_state):
        return 0.0
    elif np.any(new_state < 0.0):
        return -1000.0
    elif is_goal(new_state, goal_state):
        return 100.0
    else:
        return -0.1
	
class PetriEnv(gymnasium.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, json_obj):
        super(PetriEnv, self).__init__()

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
                    self.initial_marking[i][0] = json_obj["initial_marking"][place]
                except:
                    self.initial_marking[i][0] = 0

            self.goal_state[i][0] = 1 if json_obj["places"][place]["tokens"] == "sink" and "🗑️" not in json_obj["places"][place]["name"] else 0
        
        self.C = np.empty((self.num_places, self.num_transitions))
        # Input incedence
        self.iC = np.empty((self.num_places, self.num_transitions))
        # Output incedence
        self.oC = np.empty((self.num_places, self.num_transitions))

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

    def step(self, action):
        a = np.asarray([[0 if action != j else 1] for j in range(self.num_transitions)])
        self.previous_state = self.marking.copy()
        self.marking = self.marking + np.dot(self.C, a)
        
        tmp_rwd = reward_value(self.previous_state, self.marking, self.goal_state)

        return self.marking, tmp_rwd, is_goal(self.marking, self.goal_state), False, {}

    def reset(self, seed=0, options={}):
        self.done = False
        self.marking = self.initial_marking.copy()
        return self.marking, {}  # reward, done, info can't be included


    def valid_action_mask(self):
        valid_actions = [True for _ in range(self.num_transitions)]
        for i in range(self.num_transitions):
            column = np.empty((self.num_places, 1))
            for j in range(self.num_places):
                if self.marking[j][0] + self.iC[j][i] < 0:
                    valid_actions[i] = False
        return valid_actions