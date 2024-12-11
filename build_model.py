import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
import os
from fullcostenv import FullCostEnv
from datetime import datetime
from constants import *
from utils import *
import argparse

# Rectify the numpy versions
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def run(arguments):
    # Get a start time to judge each phase's runtime
    start = datetime.now()

    # Masking function, call the environment's masking function
    def mask_fn(env: gym.Env) -> np.ndarray:
        return env.valid_action_mask()

    # Verify the model output directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Verify the log output directory exists
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Determine which input json file to use 
    f = FILENAME # default "cost_net.json"
    if arguments.input_file is not None:
        f = arguments.input_file

    # Load petrinet data from json (transitions, places)
    [json_obj, weights, json_task, _targets, _primitives, json_agents] = LOAD_JOB_FILE(f)

    # Determine naming scheme for model output
    outputFilename = ""
    if arguments.prepend is not None:
        outputFilename += arguments.prepend + "-"
    outputFilename += f.replace(".json", "")

    # Create the 3 training environments
    fullCostTrainingEnv = FullCostEnv(json_obj, weights, json_task, json_agents)

    # Reset and mask each environment
    fullCostTrainingEnv = ActionMasker(fullCostTrainingEnv, mask_fn)  # Wrap to enable masking

    # Set model to first (deadlock) environment
    if arguments.base_model is None:
        if arguments.use_tensorboard:
            model = MaskablePPO(MaskableActorCriticPolicy, fullCostTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO(MaskableActorCriticPolicy, fullCostTrainingEnv, device="auto")
    else:
        if arguments.use_tensorboard:
            model = MaskablePPO.load(arguments.base_model, fullCostTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO.load(arguments.base_model, fullCostTrainingEnv, device="auto")


    # Train on the actual environment after we've learned to avoid deadlock scenarios
    iters = 0
    while iters < arguments.iters:
        iters += 1
        model.learn(total_timesteps=FULL_COST_TIMESTEPS)
        if not arguments.chtc and FULL_COST_ITERATION_SAVE_INTERVAL > -1 and iters % FULL_COST_ITERATION_SAVE_INTERVAL == 0:
            model.save(f"{models_dir}/{outputFilename}-full-cost-{iters}")
        if iters % PRINTING_INTERVAL == 0:
            print(f"Fullcost-{iters}")

    
    if not arguments.chtc:
        model.save(f"{models_dir}/{outputFilename}-Full-Cost-final-{arguments.process}")
    else:
        model.save(f"{outputFilename}-Full-Cost-final-{arguments.process}")

    # Get ending time of the full-cost training
    ppo_time = datetime.now()

    # Print out times
    print("TOTAL TIME: {0}".format(datetime.now() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=None, help="")
    parser.add_argument("--base-model", type=str, default=None, help="")
    parser.add_argument("--prepend", type=str, default=None, help="")
    parser.add_argument("--process", type=int, default=0, help="")
    parser.add_argument("--iters", type=int, default=MAX_FULL_COST_ITERATIONS, help="")
    parser.add_argument("--use-tensorboard", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--chtc", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    args = parser.parse_args()

    run(args)