import numpy as np
import json

# Evaluate network to determine whether the goal state(s) has been satisfied
def IS_GOAL(marking, goal_state):
    places = np.where(goal_state == 1)[0]
    for i in places:
        if marking[i][0] < 1:
            return False
    return True

def LOAD_JOB_FILE(filename):
    with open(filename, encoding='utf-8') as fh:
        json_obj = json.load(fh)
        return [json_obj["costNet"], json_obj["weights"], json_obj["tasks"]]