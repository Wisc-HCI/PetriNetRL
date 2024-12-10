import numpy as np
import json

# Evaluate network to determine whether the goal state(s) has been satisfied
def IS_GOAL(marking, goal_state):
    places = np.where(goal_state == 1)[0]
    for i in places:
        if marking[i][0] < 1:
            return False
    return True

def IS_INVALID_STATE(marking):
    return np.any(marking < 0.0)

def ALL_AGENTS_DISCARDED(marking, discard_locations):
    # Iterate over all places in the petrinet where agents are discarded (1 for each agent)
    for i in discard_locations:
        # Check if the agent discard location is empty (if so, agent hasn't been discarded)
        if marking[i][0] == 0:
            return False
        
    return True

def LOAD_JOB_FILE(filename):
    with open(filename, encoding='utf-8') as fh:
        json_obj = json.load(fh)
        return [json_obj["costNet"], json_obj["weights"], json_obj["tasks"], json_obj["targets"], json_obj["primitives"], json_obj["agents"]]