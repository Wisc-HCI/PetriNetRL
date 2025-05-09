import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from fullcostenv import FullCostEnv
from constants import *
from utils import *
import csv
import argparse
import sys
import random
import re
import os

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
    if arguments.input_file is not None:
        f = arguments.input_file
    
    # Determine naming scheme for model output
    outputFilename = f.replace(".json", "") + "-output.csv"
    if arguments.output is not None:
        outputFilename = arguments.output

    if "/" not in f:
        primitiveOutputFilename = "primitive-" + f.replace(".json", "") + "-output.csv"
    else:
        folder_path = f.split("/")
        folders = "/".join(folder_path[:len(folder_path)-1])
        primitiveOutputFilename = folders + "/primitive-" + folder_path[len(folder_path)-1].replace(".json", "") + "-output.csv"
    if arguments.output is not None:
        if "/" not in arguments.output:
            primitiveOutputFilename = "primitive-" + arguments.output
        else:
            folder_path = arguments.output.split("/")
            folders = "/".join(folder_path[:len(folder_path)-1])
            primitiveOutputFilename = folders + "/primitive-" + folder_path[len(folder_path)-1].replace(".json", "") + "-output.csv"

    # Load petrinet data from json (transitions, places)
    [json_obj, weights, json_task, targets_obj, primitives_obj, json_agents] = LOAD_JOB_FILE(f)

    # Setup evaluation environment
    env = FullCostEnv(json_obj, weights, json_task, targets_obj, json_agents)

    # Reset and mask environment
    env.reset(0, {})
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    
    # Setup tracking
    best_action_sequence = []
    best_reward = -sys.maxsize - 1

    # Load model for evaluation
    model = None
    model_files = []
    if arguments.model is not None:
        model_files = [arguments.model]
    elif arguments.model_str is not None and arguments.model_dir is not None:
        input_split = arguments.input_file.split("/")
        escaped_pattern = re.escape(arguments.model_str).replace("FF", input_split[len(input_split)-1].replace(".json", "")).replace("XX", r"\d+")
        pattern = re.compile(f"^{escaped_pattern}$")
        model_files = ["/".join([arguments.model_dir, f]) for f in os.listdir(arguments.model_dir) if pattern.match(f)]
    else:
        print("ERROR: Unable to load model file(s).")
        exit(1)

    for model_file in model_files:
        print("Currently on: " + model_file)
        model = MaskablePPO.load(model_file, weights_only=True)
        # Reset environment and get initial observation
        obs, _info = env.reset()

        loop_count = 0
        keep_looping = True

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
                action_sequence.append((env.transition_ids[action], action, rewards, info))
                
                # If goal(s) are met or max timesteps are reached, mark as done and print
                if iteration >= arguments.n_steps or done:
                    done = True

            loop_count += 1
            obs, _info = env.reset()
            
            if len(best_action_sequence) == 0 or cummulative_reward > best_reward:
                best_action_sequence = action_sequence.copy()
                best_reward = cummulative_reward
            
            if loop_count > arguments.n_samples:
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
                             "Target",
                             "Weight",
                             "Force",
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

        for (transition_id, action, reward, info) in action_sequence:
            transition = json_obj["transitions"][transition_id]
            costs = find_costs(transition)
            duration = transition["time"]
            agents = ";".join(info["busyAgents"])
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
            targets = ""
            weights = ""
            forces = ""
            
            # Iterate over transition metadata to extract information for CSV output
            for m in transition["metaData"]:
                if m["type"] == "primitiveAssignment":
                    try:
                        primitive = json_obj["nameLookup"][m["value"][1]]
                        primtives += primitive + ";"

                        if (primitive in ["Force", "Use", "Inspect", "Selection", "Hold", "Position"]):
                            some_prim_obj = primitives_obj[m["value"][1]]
                            targets += targets_obj[some_prim_obj["target"]]["name"] + ";"
                            weights += str(targets_obj[some_prim_obj["target"]]["weight"]) + ";"
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
                elif m["type"] == "target":
                    targets += targets_obj[m["value"]]["name"] + ";"
                    weights += str(targets_obj[m["value"]]["weight"]) + ";"
                elif m["type"] == "force":
                    forces += str(m["value"][1]) + ";"

            # TODO: keep track of time, busy/working agents, who all is assigned to a task (multiple agent split)
            csv_writer.writerow([env.transition_names[action], 
                                 is_sim_type(transition),
                                 reward, 
                                 agents, 
                                 info["startTime"], 
                                 info["endTime"], 
                                 fromStandLocation, 
                                 toStandLocation, 
                                 fromHandLocation, 
                                 toHandLocation,
                                 targets,
                                 weights,
                                 forces,
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
            currentTime = info["endTime"]

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
                             "Target",
                             "Weight",
                             "Force", 
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

        for (transition_id, action, reward, info) in action_sequence:
            uuid += 1
            transition = json_obj["transitions"][transition_id]
            costs = find_costs(transition)
            duration = transition["time"]
            agents = ";".join(info["busyAgents"])
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
            target = ""
            weight = ""
            force = ""

            primitive_dictionary = {}

            # iterate once to get all primitives
            for m in transition["metaData"]:
                if m["type"] == "primitiveAssignment":
                    try:
                        primitive_dictionary[m["value"][1]] = ["" for _ in range(28)]
                        primitive_dictionary[m["value"][1]][0] = uuid
                        primitive_dictionary[m["value"][1]][1] = env.transition_names[action]

                        primitive = json_obj["nameLookup"][m["value"][1]]
                        primitive_dictionary[m["value"][1]][2] = primitive
                        primitive_dictionary[m["value"][1]][3] = is_sim_type(transition)
                        primitive_dictionary[m["value"][1]][4] = reward
                        
                        # ... left blank for agent (for now)
                        primitive_dictionary[m["value"][1]][5] = json_agents[m["value"][0]]["name"]

                        primitive_dictionary[m["value"][1]][6] = info["startTime"]
                        primitive_dictionary[m["value"][1]][7] = info["endTime"]
                        # ....
                        if (primitive in ["Force", "Use", "Inspect", "Selection", "Hold", "Position"]):
                            try:
                                some_prim_obj = primitives_obj[m["value"][1]]
                                primitive_dictionary[m["value"][1]][12] = targets_obj[some_prim_obj["target"]]["name"]
                                primitive_dictionary[m["value"][1]][13] = targets_obj[some_prim_obj["target"]]["weight"]
                            except:
                                a = 1
                        # ....
                        primitive_dictionary[m["value"][1]][15] = costs[0]
                        primitive_dictionary[m["value"][1]][16] = costs[1]
                        primitive_dictionary[m["value"][1]][17] = costs[2]
                        primitive_dictionary[m["value"][1]][18] = costs[3]
                        primitive_dictionary[m["value"][1]][19] = costs[4]
                    except:
                        a = 1


            # Iterate over transition metadata to extract information for CSV output
            for m in transition["metaData"]:
                if m["type"] == "mVC":
                    primitive_dictionary[m["value"][0]][20] = str(m["value"][1])
                elif m["type"] == "verticalHandTravelDistance":
                    primitive_dictionary[m["value"][0]][21] = str(m["value"][1])
                elif m["type"] == "horizontalHandTravelDistance":
                    primitive_dictionary[m["value"][0]][22] = str(m["value"][1])
                elif m["type"] == "standTravelDistance":
                    primitive_dictionary[m["value"][0]][23] = str(m["value"][1])
                elif m["type"] == "reachDistance":
                    primitive_dictionary[m["value"][0]][24] = str(m["value"][1])
                elif m["type"] == "handDistanceToFloor":
                    primitive_dictionary[m["value"][0]][25] = str(m["value"][1])
                if m["type"] == "isOneHanded":
                    primitive_dictionary[m["value"][0]][26] = "True" if m["value"][1] else "False"
                elif m["type"] == "isHandWork":
                    primitive_dictionary[m["value"][0]][27] = "True" if m["value"][1] else "False"
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
                elif m["type"] == "target":
                    target = targets_obj[m["value"]]["name"]
                    weight = targets_obj[m["value"]]["weight"]
                elif m["type"] == "force":
                    primitive_dictionary[m["value"][0]][14] = str(m["value"][1])

            if len(primitive_dictionary.keys()) == 0:
                primitive_dictionary[""] = [uuid,
                                            env.transition_names[action],
                                            "",
                                            is_sim_type(transition),
                                            reward,
                                            agents,
                                            info["startTime"], 
                                            info["endTime"], 
                                            fromStandLocation,
                                            toStandLocation,
                                            fromHandLocation,
                                            toHandLocation,
                                            target,
                                            weight,
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
                                            "",
                                            ]

            for key in primitive_dictionary:
                primitive_dictionary[key][8] = fromStandLocation
                primitive_dictionary[key][9] = toStandLocation
                primitive_dictionary[key][10] = fromHandLocation
                primitive_dictionary[key][11] = toHandLocation
                if primitive_dictionary[key][12] == "":
                    primitive_dictionary[key][12] = target
                    primitive_dictionary[key][13] = weight
                csv_writer.writerow(primitive_dictionary[key])

            currentTime = info["endTime"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=None, help="")
    parser.add_argument("--model", type=str, default=None, help="")
    parser.add_argument("--model-str", type=str, default=None, help="")
    parser.add_argument("--model-dir", type=str, default=None, help="")
    parser.add_argument("--output", type=str, default=None, help="")
    parser.add_argument("--n-steps", type=int, default=MAX_TESTING_TIMESTEPS, help="")
    parser.add_argument("--n-samples", type=int, default=0, help="")
    args = parser.parse_args()

    run(args)