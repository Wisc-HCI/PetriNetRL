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
import time
import json
from datetime import datetime
from constants import *
import argparse

# Rectify the numpy versions
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def run(filename):
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
    if filename is not None:
        f = filename
    # Load petrinet data from json (transitions, places)
    with open(f, encoding='utf-8') as fh:
        json_obj = json.load(fh)

    # Determine naming scheme for model output
    outputFilename = f.replace(".json", "")

    # Create the 3 training environments
    deadlockTrainingEnv = DeadlockEnv(json_obj)
    explorationTrainingEnv = ExplorationEnv(json_obj)
    fullCostTrainingEnv = FullCostEnv(json_obj)

    # Reset and mask each environment
    deadlockTrainingEnv.reset(0, {})
    explorationTrainingEnv.reset(0, {})
    deadlockTrainingEnv = ActionMasker(deadlockTrainingEnv, mask_fn)  # Wrap to enable masking
    explorationTrainingEnv = ActionMasker(explorationTrainingEnv, mask_fn)  # Wrap to enable masking
    fullCostTrainingEnv = ActionMasker(fullCostTrainingEnv, mask_fn)  # Wrap to enable masking

    # Set model to first (deadlock) environment
    model = MaskablePPO(MaskableActorCriticPolicy, deadlockTrainingEnv, verbose=1, tensorboard_log=logdir, device="auto")

    # Check if deadlock training is active
    if DEADLOCK_TRAINING:
        # Train model in the deadlock environment
        iters = 0
        while iters < MAX_DEADLOCK_ITERATIONS:
            iters += 1
            model.learn(total_timesteps=DEADLOCK_TIMESTEPS)
            if iters % DEADLOCK_ITERATION_SAVE_INTERVAL == 0:
                model.save(f"{models_dir}/{outputFilename}-deadlock-{iters}")

        # After training, save and load the model to change environments for the next round of training
        model.save(f"{models_dir}/Deadlock-final")

        # If the next phase is exploration, load the model for the second environment, otherwise the third (full-cost environment)
        if EXPLORATION_TRAINING:
            model = MaskablePPO.load(f"{models_dir}/Deadlock-final", explorationTrainingEnv, verbose=1, tensorboard_log=logdir, device="auto")
        else:
            model = MaskablePPO.load(f"{models_dir}/Deadlock-final", fullCostTrainingEnv, verbose=1, tensorboard_log=logdir, device="auto")
    else:
        # Assign some base to start from
        # TODO: swap this to a check for prior base or just starting in explorationTrainingEnv with no base
        model = MaskablePPO.load(f"{models_dir}/Deadlock-5", explorationTrainingEnv, verbose=1, tensorboard_log=logdir, device="auto")

    # Get ending time of the deadlock training
    deadlock_time = datetime.now()

    # Determine if exploration training is active
    if EXPLORATION_TRAINING:
        # Train model in the exploration environment
        iters = 0
        while iters < MAX_EXPLORATION_ITERATIONS:
            iters += 1
            model.learn(total_timesteps=EXPLORATION_TIMESTEPS)
            if iters % EXPLORATION_ITERATION_SAVE_INTERVAL == 0:
                model.save(f"{models_dir}/{outputFilename}-exploration-{iters}")

        # After training, save and load the model to change environments for the next round of training
        model.save(f"{models_dir}/Exploration-final")

        # Load model for the third environment (full-cost environment)
        model = MaskablePPO.load(f"{models_dir}/Exploration-final", fullCostTrainingEnv, verbose=1, tensorboard_log=logdir, device="auto")
    elif not DEADLOCK_TRAINING:
        # If deadlock training was not active (and exploration training isn't active) load some base model
        # TODO: swap this to a check for prior base or just starting in fullCostTrainingEnv with no base
        model = MaskablePPO.load(f"{models_dir}/Deadlock-10", fullCostTrainingEnv, verbose=1, tensorboard_log=logdir, device="auto")

    # Get ending time of the exploration training
    exploration_time = datetime.now()

    # Check if third environment training is active
    if FULL_COST_TRAINING:
        # Train on the actual environment after we've learned to avoid deadlock scenarios
        iters = 0
        while iters < MAX_FULL_COST_ITERATIONS:
            iters += 1
            model.learn(total_timesteps=FULL_COST_TIMESTEPS)
            if iters % FULL_COST_ITERATION_SAVE_INTERVAL == 0:
                model.save(f"{models_dir}/{outputFilename}-full-cost-{iters}")
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
    parser.add_argument("--file", type=str, default=None, help="")
    args = parser.parse_args()

    run(args.file)