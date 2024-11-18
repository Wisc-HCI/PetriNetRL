import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from fullcostenv import FullCostEnv
from explorationenv import ExplorationEnv
from deadlockenv import DeadlockEnv
from constants import *
from utils import *
import csv
import argparse
import sys

# Inspect transition metadata to determine interaction phase type (setup or simulation)
def is_sim_type(transition):
    for obj in transition["metaData"]:
        if obj["type"] == "simulation":
            return "Simulation"
    return "Setup"

# Inspect transition metadata to determine ergonomic and monetary costs
def find_costs(transition):
    costs = [0] * 5
    for obj in transition["metaData"]:
        if obj["type"] == "ergoHand":
            costs[0] += obj["value"][1]
        if obj["type"] == "ergoArm":
            costs[1] += obj["value"][1]
        if obj["type"] == "ergoShoulder":
            costs[2] += obj["value"][1]
        if obj["type"] == "ergoWholeBody":
            costs[3] += obj["value"][1]
    for obj in transition["cost"]:
        if obj["category"] == "monetary":
            costs[4] += obj["value"]

    return costs

# Masking function, call the environment's masking function
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


def run(arguments):
    # Determine which input json file to use 
    f = FILENAME # default "cost_net.json"
    if arguments.inputfile is not None:
        f = arguments.inputfile
    
    # Determine naming scheme for model output
    outputFilename = f.replace(".json", "") + "-output.csv"
    if arguments.output is not None:
        outputFilename = arguments.output

    primitiveOutputFilename = "primitive-" + f.replace(".json", "") + "-output.csv"
    if arguments.output is not None:
        primitiveOutputFilename = "primitive-" + arguments.output

    # Load petrinet data from json (transitions, places)
    [json_obj, weights, json_task] = LOAD_JOB_FILE(f)

    # Setup evaluation environment
    if (arguments.useExploreEnv):
        print("Using Exploration Env")
        env = ExplorationEnv(json_obj, json_task)
    else:
        print("Using Full Cost Env")
        env = FullCostEnv(json_obj, weights, json_task)

    # Reset and mask environment
    env.reset(0, {})
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # Load model for evaluation
    model = None
    if arguments.model is not None:
        model = MaskablePPO.load(arguments.model, weights_only=True)
    else:
        model = MaskablePPO.load("models/PetriNet-PPO/Exploration-final.zip")

    # Reset environment and get initial observation
    obs, _info = env.reset()

    # Setup tracking
    loop_count = 0
    keep_looping = True

    best_action_sequence = []
    best_reward = -sys.maxsize - 1

    while keep_looping:
        cummulative_reward = 0.0
        iteration = 0
        action_sequence = []
        done = False
        # Iterate until model finds the goal(s) or max timesteps are hit
        while not done:
            # Increase iteration count
            iteration += 1
            
            # Determine best action from current state
            action, _states = model.predict(obs, action_masks=mask_fn(env))
            
            # Step the model based on selected action
            obs, rewards, done, shortcut, info = env.step(action)

            # Update rewards
            cummulative_reward += rewards
            
            # Add selected action to sequence
            action_sequence.append((env.transition_ids[action], action, rewards))
            
            # If goal(s) are met or max timesteps are reached, mark as done and print
            if iteration >= MAX_TESTING_TIMESTEPS or done:
                done = True
                # print("reward", rewards)
                # print("cumulative reward", cummulative_reward)
                # print("is goal met? + {0}".format(IS_GOAL(obs, env.goal_state)))
                # print("Ending due to iteration cap or done flag")
                # print("iteration", iteration, "Max", MAX_TESTING_TIMESTEPS)

        loop_count += 1
        obs, _info = env.reset()

        if len(action_sequence) < len(best_action_sequence) or len(best_action_sequence) == 0:
            best_action_sequence = action_sequence.copy()
            best_reward = cummulative_reward
        elif len(action_sequence) == len(best_action_sequence) and cummulative_reward > best_reward:
            best_action_sequence = action_sequence.copy()
            best_reward = cummulative_reward
        
        if loop_count > arguments.maxretries or len(action_sequence) < arguments.targetsteps:
            keep_looping = False

    action_sequence = best_action_sequence.copy()
    print(len(action_sequence), best_reward)
    currentTime = 0

    # Write output to CSV
    with open(outputFilename, "w+", newline='') as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(["Action", 
                             "Type", 
                             "Reward", 
                             "Agent Assigned To Task", 
                             "Start Time (s)", 
                             "End Time (s)", 
                             "From Standing Location", 
                             "To Standing Location", 
                             "From Hand Location", 
                             "To Hand Location", 
                             "Hand Cost", 
                             "Arm Cost", 
                             "Shoulder Cost", 
                             "Whole Body Cost", 
                             "Monetary Cost", 
                             "Primitives", 
                             "MVC", 
                             "Vertical Hand Distance Traveled", 
                             "Horizontal Hand Distance Traveled", 
                             "Stand Distance Traveled", 
                             "Reach Distance", 
                             "Hand To Floor Distance", 
                             "Is One Handed", 
                             "Is Hand Work"])

        for (transition_id, action, reward) in action_sequence:
            transition = json_obj["transitions"][transition_id]
            costs = find_costs(transition)
            duration = transition["time"]
            primtives = ""
            mvcs = ""
            isOneHanded = ""
            isHandWork = ""
            standDistanceTraveled = ""
            horizontalHandDistanceTraveled = ""
            verticalHandDistanceTraveled = ""
            reachDistance = ""
            distanceFromHandToFloor = ""
            a = 0
            fromHandLocation = ""
            toHandLocation = ""
            fromStandLocation = ""
            toStandLocation = ""
            
            # Iterate over transition metadata to extract information for CSV output
            for m in transition["metaData"]:
                if m["type"] == "primitiveAssignment":
                    try:
                        primtives += json_obj["nameLookup"][m["value"][1]] + ";"
                    except:
                        a = 1
                elif m["type"] == "isOneHanded":
                    isOneHanded += "True" if m["value"][1] else "False" + ";"
                elif m["type"] == "isHandWork":
                    isHandWork += "True" if m["value"][1] else "False" + ";"
                elif m["type"] == "mVC":
                    mvcs += str(m["value"][1]) + ";"
                elif m["type"] == "horizontalHandTravelDistance":
                    horizontalHandDistanceTraveled += str(m["value"][1]) + ";"
                elif m["type"] == "verticalHandTravelDistance":
                    verticalHandDistanceTraveled += str(m["value"][1]) + ";"
                elif m["type"] == "standTravelDistance":
                    standDistanceTraveled += str(m["value"][1]) + ";"
                elif m["type"] == "reachDistance":
                    reachDistance += str(m["value"][1]) + ";"
                elif m["type"] == "handDistanceToFloor":
                    distanceFromHandToFloor += str(m["value"][1]) + ";"
                elif m["type"] == "standing":
                    fromStandLocation = json_obj["nameLookup"][m["value"][0]]
                    toStandLocation = fromStandLocation
                elif m["type"] == "hand":
                    fromHandLocation = json_obj["nameLookup"][m["value"][0]]
                    toHandLocation = fromHandLocation
                elif m["type"] == "fromHandPOI":
                    fromHandLocation = json_obj["nameLookup"][m["value"][0]]
                elif m["type"] == "toHandPOI":
                    toHandLocation = json_obj["nameLookup"][m["value"][0]]
                elif m["type"] == "fromStandingPOI":
                    fromStandLocation = json_obj["nameLookup"][m["value"][0]]
                elif m["type"] == "toStandingPOI":
                    toStandLocation = json_obj["nameLookup"][m["value"][0]]

            # TODO: keep track of time, busy/working agents, who all is assigned to a task (multiple agent split)
            csv_writer.writerow([env.transition_names[action], 
                                 is_sim_type(transition),
                                 reward, 
                                 "", 
                                 currentTime, 
                                 currentTime+duration, 
                                 fromStandLocation, 
                                 toStandLocation, 
                                 fromHandLocation, 
                                 toHandLocation, 
                                 costs[0], 
                                 costs[1], 
                                 costs[2], 
                                 costs[3], 
                                 costs[4], 
                                 primtives, 
                                 mvcs, 
                                 verticalHandDistanceTraveled,
                                 horizontalHandDistanceTraveled, 
                                 standDistanceTraveled, 
                                 reachDistance,
                                 distanceFromHandToFloor,
                                 isOneHanded, 
                                 isHandWork
                                 ])
            currentTime += duration

    currentTime = 0
    uuid = 0
    # Write output to primitives CSV
    with open(primitiveOutputFilename, "w+", newline='') as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(["Unique ID",
                             "Action",
                             "Primitive",
                             "Type",
                             "Reward", 
                             "Agent Assigned To Task", 
                             "Start Time (s)", 
                             "End Time (s)", 
                             "From Standing Location", 
                             "To Standing Location", 
                             "From Hand Location", 
                             "To Hand Location", 
                             "Hand Cost", 
                             "Arm Cost", 
                             "Shoulder Cost", 
                             "Whole Body Cost", 
                             "Monetary Cost",
                             "MVC", 
                             "Vertical Hand Distance Traveled", 
                             "Horizontal Hand Distance Traveled", 
                             "Stand Distance Traveled", 
                             "Reach Distance", 
                             "Hand To Floor Distance", 
                             "Is One Handed", 
                             "Is Hand Work"])

        for (transition_id, action, reward) in action_sequence:
            uuid += 1
            transition = json_obj["transitions"][transition_id]
            costs = find_costs(transition)
            duration = transition["time"]
            primtives = ""
            mvcs = ""
            isOneHanded = ""
            isHandWork = ""
            standDistanceTraveled = ""
            horizontalHandDistanceTraveled = ""
            verticalHandDistanceTraveled = ""
            reachDistance = ""
            distanceFromHandToFloor = ""
            a = 0
            fromHandLocation = ""
            toHandLocation = ""
            fromStandLocation = ""
            toStandLocation = ""

            primitive_dictionary = {}

            # iterate once to get all primitives
            for m in transition["metaData"]:
                if m["type"] == "primitiveAssignment":
                    try:
                        primitive_dictionary[m["value"][1]] = ["" for _ in range(25)]
                        primitive_dictionary[m["value"][1]][0] = uuid
                        primitive_dictionary[m["value"][1]][1] = env.transition_names[action]
                        primitive_dictionary[m["value"][1]][2] = json_obj["nameLookup"][m["value"][1]]
                        primitive_dictionary[m["value"][1]][3] = is_sim_type(transition)
                        primitive_dictionary[m["value"][1]][4] = reward
                        # ... left blank for agent (for now)
                        primitive_dictionary[m["value"][1]][6] = currentTime
                        primitive_dictionary[m["value"][1]][7] = currentTime+duration
                        # ....
                        primitive_dictionary[m["value"][1]][12] = costs[0]
                        primitive_dictionary[m["value"][1]][13] = costs[1]
                        primitive_dictionary[m["value"][1]][14] = costs[2]
                        primitive_dictionary[m["value"][1]][15] = costs[3]
                        primitive_dictionary[m["value"][1]][16] = costs[4]
                    except:
                        a = 1


            # Iterate over transition metadata to extract information for CSV output
            for m in transition["metaData"]:
                if m["type"] == "mVC":
                    primitive_dictionary[m["value"][0]][17] = str(m["value"][1])
                elif m["type"] == "verticalHandTravelDistance":
                    primitive_dictionary[m["value"][0]][18] = str(m["value"][1])
                elif m["type"] == "horizontalHandTravelDistance":
                    primitive_dictionary[m["value"][0]][19] = str(m["value"][1])
                elif m["type"] == "standTravelDistance":
                    primitive_dictionary[m["value"][0]][20] = str(m["value"][1])
                elif m["type"] == "reachDistance":
                    primitive_dictionary[m["value"][0]][21] = str(m["value"][1])
                elif m["type"] == "handDistanceToFloor":
                    primitive_dictionary[m["value"][0]][22] = str(m["value"][1])
                if m["type"] == "isOneHanded":
                    primitive_dictionary[m["value"][0]][23] = "True" if m["value"][1] else "False"
                elif m["type"] == "isHandWork":
                    primitive_dictionary[m["value"][0]][24] = "True" if m["value"][1] else "False"
                elif m["type"] == "standing":
                    fromStandLocation = json_obj["nameLookup"][m["value"][0]]
                    toStandLocation = fromStandLocation
                elif m["type"] == "hand":
                    fromHandLocation = json_obj["nameLookup"][m["value"][0]]
                    toHandLocation = fromHandLocation
                elif m["type"] == "fromHandPOI":
                    fromHandLocation = json_obj["nameLookup"][m["value"][0]]
                elif m["type"] == "toHandPOI":
                    toHandLocation = json_obj["nameLookup"][m["value"][0]]
                elif m["type"] == "fromStandingPOI":
                    fromStandLocation = json_obj["nameLookup"][m["value"][0]]
                elif m["type"] == "toStandingPOI":
                    toStandLocation = json_obj["nameLookup"][m["value"][0]]

            if len(primitive_dictionary.keys()) == 0:
                primitive_dictionary[""] = [uuid,
                                            env.transition_names[action],
                                            "",
                                            is_sim_type(transition),
                                            reward,
                                            "",
                                            currentTime,
                                            currentTime+duration,
                                            fromStandLocation,
                                            toStandLocation,
                                            fromHandLocation,
                                            toHandLocation,
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            ]

            for key in primitive_dictionary:
                primitive_dictionary[key][8] = fromStandLocation
                primitive_dictionary[key][9] = toStandLocation
                primitive_dictionary[key][10] = fromHandLocation
                primitive_dictionary[key][11] = toHandLocation
                csv_writer.writerow(primitive_dictionary[key])

            currentTime += duration



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", type=str, default=None, help="")
    parser.add_argument("--model", type=str, default=None, help="")
    parser.add_argument("--output", type=str, default=None, help="")
    parser.add_argument("--targetsteps", type=int, default=1000, help="")
    parser.add_argument("--maxretries", type=int, default=0, help="")
    parser.add_argument("--useExploreEnv", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    args = parser.parse_args()

    run(args)