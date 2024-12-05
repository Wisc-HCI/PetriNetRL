from constants import *
from utils import *
import argparse
import sys

def verifyJobFile(json_obj):

    passMVC = True
    maxMVC = 0
    maxTransition = ""
    for i, place in enumerate(json_obj["transitions"]):
        for data in json_obj["transitions"][place]["metaData"]:
            if data["type"] == "mVC" and data["value"][1] > maxMVC:
                maxMVC = data["value"][1]
                maxTransition = place
            if data["type"] == "mVC" and data["value"][1] > 1.5:
                passMVC = False
    print(maxMVC, maxTransition)
    assert passMVC

def verfiyTransitionCosts(json_obj, weights_obj):
    weights = [weights_obj[ERGO_KEY], weights_obj[MONETARY_KEY]]
    transition_costs = [[0, 0] for _ in range(len(json_obj["transitions"]))]
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
                multiplier = weights[0]
            else:
                multiplier = weights[1]

            # Apply weighting to costs
            if c["frequency"] == ONE_TIME_KEY:
                one_time_cost += multiplier * c["value"]
            else:
                extrapolated_cost += multiplier * c["value"]

        transition_costs[i] = [one_time_cost, extrapolated_cost]

    # Find min/max of the extrapolated and one time costs
    maxValue = -sys.maxsize - 1
    minValue = sys.maxsize
    maxValue2 = -sys.maxsize - 1
    minValue2 = sys.maxsize
    for i in range(len(transition_costs)):
        if transition_costs[i][0] >= 0:
            maxValue = max(maxValue, transition_costs[i][0])
            minValue = min(minValue, transition_costs[i][0])

        if transition_costs[i][1] >= 0:
            maxValue2 = max(maxValue2, transition_costs[i][1])
            minValue2 = min(minValue2, transition_costs[i][1])

    print(minValue, maxValue, minValue2, maxValue2)

    # Normalize (0-1) based on the found min/maxs
    for i in range(len(transition_costs)):
        if transition_costs[i][0] > 0:
            transition_costs[i][0] = (transition_costs[i][0]-minValue) / (maxValue - minValue)
        if transition_costs[i][1] > 0:
            transition_costs[i][1] = (transition_costs[i][1]-minValue2) / (maxValue2 - minValue2)
        if abs(transition_costs[i][1]) > 1 or abs(transition_costs[i][0]) > 1:
            print(transition_costs[i])
        
    return

def getAllOneTimeCosts(json_obj):
    for i, place in enumerate(json_obj["transitions"]):
        for data in json_obj["transitions"][place]["cost"]:
            if data["frequency"] == "once":
                print(place, data["value"])

def run(arguments):
    f = FILENAME # default "cost_net.json"
    if arguments.inputfile is not None:
        f = arguments.inputfile

    # Load petrinet data from json (transitions, places)
    [json_obj, _weights, _json_task, _targets, _primitives, _agents] = LOAD_JOB_FILE(f)

    if not arguments.disableMVCVerify:
        verifyJobFile(json_obj)

    if arguments.printOneTimeCosts:
        getAllOneTimeCosts(json_obj)

    verfiyTransitionCosts(json_obj, _weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", type=str, default=None, help="")
    parser.add_argument("--disableMVCVerify", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--printOneTimeCosts", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    args = parser.parse_args()

    run(args)