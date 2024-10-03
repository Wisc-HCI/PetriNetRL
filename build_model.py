import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
import os
from fullcostenv import FullCostEnv
from deadlockenv import DeadlockEnv
from explorationenv import ExplorationEnv
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
    if arguments.inputfile is not None:
        f = arguments.inputfile

    # Load petrinet data from json (transitions, places)
    [json_obj, weights] = LOAD_JOB_FILE(f)

    # Determine naming scheme for model output
    outputFilename = f.replace(".json", "")

    # Create the 3 training environments
    deadlockTrainingEnv = DeadlockEnv(json_obj)
    explorationTrainingEnv = ExplorationEnv(json_obj)
    fullCostTrainingEnv = FullCostEnv(json_obj, weights)

    # Reset and mask each environment
    deadlockTrainingEnv.reset(0, {})
    explorationTrainingEnv.reset(0, {})
    deadlockTrainingEnv = ActionMasker(deadlockTrainingEnv, mask_fn)  # Wrap to enable masking
    explorationTrainingEnv = ActionMasker(explorationTrainingEnv, mask_fn)  # Wrap to enable masking
    fullCostTrainingEnv = ActionMasker(fullCostTrainingEnv, mask_fn)  # Wrap to enable masking

    # Set model to first (deadlock) environment
    if arguments.baseModel is None and arguments.enableDeadlock:
        if arguments.useTensorboard:
            model = MaskablePPO(MaskableActorCriticPolicy, deadlockTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO(MaskableActorCriticPolicy, deadlockTrainingEnv, device="auto")
    elif arguments.baseModel is None and arguments.enableExploration:
        if arguments.useTensorboard:
            model = MaskablePPO(MaskableActorCriticPolicy, explorationTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO(MaskableActorCriticPolicy, explorationTrainingEnv, device="auto")
    elif arguments.baseModel is None and arguments.enableFullcost:
        if arguments.useTensorboard:
            model = MaskablePPO(MaskableActorCriticPolicy, fullCostTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO(MaskableActorCriticPolicy, fullCostTrainingEnv, device="auto")
    elif arguments.enableDeadlock:
        if arguments.useTensorboard:
            model = MaskablePPO.load(arguments.baseModel, deadlockTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO.load(arguments.baseModel, deadlockTrainingEnv, device="auto")
    elif arguments.enableExploration:
        if arguments.useTensorboard:
            model = MaskablePPO.load(arguments.baseModel, explorationTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO.load(arguments.baseModel, explorationTrainingEnv, device="auto")
    elif arguments.enableFullcost:
        if arguments.useTensorboard:
            model = MaskablePPO.load(arguments.baseModel, fullCostTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO.load(arguments.baseModel, fullCostTrainingEnv, device="auto")

    # Check if deadlock training is active
    if arguments.enableDeadlock:
        # Train model in the deadlock environment
        iters = 0
        while iters < MAX_DEADLOCK_ITERATIONS:
            iters += 1
            model.learn(total_timesteps=DEADLOCK_TIMESTEPS)
            if DEADLOCK_ITERATION_SAVE_INTERVAL > -1 and iters % DEADLOCK_ITERATION_SAVE_INTERVAL == 0:
                model.save(f"{models_dir}/{outputFilename}-deadlock-{iters}")
            elif iters % PRINTING_INTERVAL == 0:
                print(f"Deadlock-{iters}")

        # After training, save and load the model to change environments for the next round of training
        model.save(f"{models_dir}/Deadlock-final")

        # If the next phase is exploration, load the model for the second environment, otherwise the third (full-cost environment)
        if arguments.enableExploration:
            if arguments.useTensorboard:
                model = MaskablePPO.load(f"{models_dir}/Deadlock-final", explorationTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
            else:
                model = MaskablePPO.load(f"{models_dir}/Deadlock-final", explorationTrainingEnv, device="auto")
        else:
            if arguments.useTensorboard:
                model = MaskablePPO.load(f"{models_dir}/Deadlock-final", fullCostTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
            else:
                model = MaskablePPO.load(f"{models_dir}/Deadlock-final", fullCostTrainingEnv, device="auto")

    # Get ending time of the deadlock training
    deadlock_time = datetime.now()

    # Determine if exploration training is active
    if arguments.enableExploration:
        # Train model in the exploration environment
        iters = 0
        while iters < MAX_EXPLORATION_ITERATIONS:
            iters += 1
            model.learn(total_timesteps=EXPLORATION_TIMESTEPS)
            if EXPLORATION_ITERATION_SAVE_INTERVAL > -1 and iters % EXPLORATION_ITERATION_SAVE_INTERVAL == 0:
                model.save(f"{models_dir}/{outputFilename}-exploration-{iters}")
            elif iters % PRINTING_INTERVAL == 0:
                print(f"Exploration-{iters}")


        # After training, save and load the model to change environments for the next round of training
        model.save(f"{models_dir}/Exploration-final")

        # Load model for the third environment (full-cost environment)
        if arguments.useTensorboard:
            model = MaskablePPO.load(f"{models_dir}/Exploration-final", fullCostTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO.load(f"{models_dir}/Exploration-final", fullCostTrainingEnv, device="auto")

    # Get ending time of the exploration training
    exploration_time = datetime.now()

    # Check if third environment training is active
    if arguments.enableFullcost:
        # Train on the actual environment after we've learned to avoid deadlock scenarios
        iters = 0
        while iters < MAX_FULL_COST_ITERATIONS:
            iters += 1
            model.learn(total_timesteps=FULL_COST_TIMESTEPS)
            if FULL_COST_ITERATION_SAVE_INTERVAL > -1 and iters % FULL_COST_ITERATION_SAVE_INTERVAL == 0:
                model.save(f"{models_dir}/{outputFilename}-full-cost-{iters}")
            elif iters % PRINTING_INTERVAL == 0:
                print(f"Fullcost-{iters}")
        model.save(f"{models_dir}/Full-Cost-final")

    # Get ending time of the full-cost training
    ppo_time = datetime.now()

    # Print out times
    print("Deadlock TIME: {0}".format(deadlock_time - start))
    print("Exploration TIME: {0}".format(exploration_time - deadlock_time))
    print("Full-Cost TIME: {0}".format(ppo_time - exploration_time))
    print("TOTAL TIME: {0}".format(datetime.now() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", type=str, default=None, help="")
    parser.add_argument("--baseModel", type=str, default=None, help="")
    parser.add_argument("--enableDeadlock", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--enableExploration", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--enableFullcost", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--useTensorboard", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    args = parser.parse_args()

    run(args)