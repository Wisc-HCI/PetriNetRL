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
    deadlockTrainingEnv = DeadlockEnv(json_obj)
    explorationTrainingEnv = ExplorationEnv(json_obj, json_task)
    fullCostTrainingEnv = FullCostEnv(json_obj, weights, json_task, json_agents)

    # Reset and mask each environment
    deadlockTrainingEnv.reset(0, {})
    explorationTrainingEnv.reset(0, {})
    deadlockTrainingEnv = ActionMasker(deadlockTrainingEnv, mask_fn)  # Wrap to enable masking
    explorationTrainingEnv = ActionMasker(explorationTrainingEnv, mask_fn)  # Wrap to enable masking
    fullCostTrainingEnv = ActionMasker(fullCostTrainingEnv, mask_fn)  # Wrap to enable masking

    # Set model to first (deadlock) environment
    if arguments.base_model is None and arguments.enable_deadlock:
        if arguments.use_tensorboard:
            model = MaskablePPO(MaskableActorCriticPolicy, deadlockTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO(MaskableActorCriticPolicy, deadlockTrainingEnv, device="auto")
    elif arguments.base_model is None and arguments.enable_exploration:
        if arguments.use_tensorboard:
            model = MaskablePPO(MaskableActorCriticPolicy, explorationTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO(MaskableActorCriticPolicy, explorationTrainingEnv, device="auto")
    elif arguments.base_model is None and arguments.enable_full_cost:
        if arguments.use_tensorboard:
            model = MaskablePPO(MaskableActorCriticPolicy, fullCostTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO(MaskableActorCriticPolicy, fullCostTrainingEnv, device="auto")
    elif arguments.enable_deadlock:
        if arguments.use_tensorboard:
            model = MaskablePPO.load(arguments.base_model, deadlockTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO.load(arguments.base_model, deadlockTrainingEnv, device="auto")
    elif arguments.enable_exploration:
        if arguments.use_tensorboard:
            model = MaskablePPO.load(arguments.base_model, explorationTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO.load(arguments.base_model, explorationTrainingEnv, device="auto")
    elif arguments.enable_full_cost:
        if arguments.use_tensorboard:
            model = MaskablePPO.load(arguments.base_model, fullCostTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO.load(arguments.base_model, fullCostTrainingEnv, device="auto")

    # Check if deadlock training is active
    if arguments.enable_deadlock:
        # Train model in the deadlock environment
        iters = 0
        while iters < arguments.deadlock_iters:
            iters += 1
            model.learn(total_timesteps=DEADLOCK_TIMESTEPS)
            if not arguments.chtc and DEADLOCK_ITERATION_SAVE_INTERVAL > -1 and iters % DEADLOCK_ITERATION_SAVE_INTERVAL == 0:
                model.save(f"{models_dir}/{outputFilename}-deadlock-{iters}")
            if iters % PRINTING_INTERVAL == 0:
                print(f"Deadlock-{iters}")

        # After training, save and load the model to change environments for the next round of training
        if not arguments.chtc:
            model.save(f"{models_dir}/{outputFilename}-Deadlock-final-{arguments.process}")
        else:
            model.save(f"{outputFilename}-Deadlock-final-{arguments.process}")

        # If the next phase is exploration, load the model for the second environment, otherwise the third (full-cost environment)
        base = f"{models_dir}/"
        if arguments.chtc:
            base = ""
        if arguments.enable_exploration:
            if arguments.use_tensorboard:
                model = MaskablePPO.load(f"{base}{outputFilename}-Deadlock-final-{arguments.process}", explorationTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
            else:
                model = MaskablePPO.load(f"{base}{outputFilename}-Deadlock-final-{arguments.process}", explorationTrainingEnv, device="auto")
        else:
            if arguments.use_tensorboard:
                model = MaskablePPO.load(f"{base}{outputFilename}-Deadlock-final-{arguments.process}", fullCostTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
            else:
                model = MaskablePPO.load(f"{base}{outputFilename}-Deadlock-final-{arguments.process}", fullCostTrainingEnv, device="auto")

    # Get ending time of the deadlock training
    deadlock_time = datetime.now()

    # Determine if exploration training is active
    if arguments.enable_exploration:
        # Train model in the exploration environment
        iters = 0
        while iters < arguments.explore_iters:
            iters += 1
            model.learn(total_timesteps=EXPLORATION_TIMESTEPS)
            if not arguments.chtc and EXPLORATION_ITERATION_SAVE_INTERVAL > -1 and iters % EXPLORATION_ITERATION_SAVE_INTERVAL == 0:
                model.save(f"{models_dir}/{outputFilename}-exploration-{iters}")
            if iters % PRINTING_INTERVAL == 0:
                print(f"Exploration-{iters}")


        # After training, save and load the model to change environments for the next round of training
        if not arguments.chtc:
            model.save(f"{models_dir}/{outputFilename}-Exploration-final-{arguments.process}")
        else:
            model.save(f"{outputFilename}-Exploration-final-{arguments.process}")

        # Load model for the third environment (full-cost environment)
        base = f"{models_dir}/"
        if arguments.chtc:
            base = ""
        if arguments.use_tensorboard:
            model = MaskablePPO.load(f"{base}{outputFilename}-Exploration-final-{arguments.process}", fullCostTrainingEnv, verbose=1, tensorbord_log=logdir, device="auto")
        else:
            model = MaskablePPO.load(f"{base}{outputFilename}-Exploration-final-{arguments.process}", fullCostTrainingEnv, device="auto")

    # Get ending time of the exploration training
    exploration_time = datetime.now()

    # Check if third environment training is active
    if arguments.enable_full_cost:
        # Train on the actual environment after we've learned to avoid deadlock scenarios
        iters = 0
        while iters < arguments.fullcost_iters:
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
    print("Deadlock TIME: {0}".format(deadlock_time - start))
    print("Exploration TIME: {0}".format(exploration_time - deadlock_time))
    print("Full-Cost TIME: {0}".format(ppo_time - exploration_time))
    print("TOTAL TIME: {0}".format(datetime.now() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=None, help="")
    parser.add_argument("--base-model", type=str, default=None, help="")
    parser.add_argument("--prepend", type=str, default=None, help="")
    parser.add_argument("--process", type=int, default=0, help="")
    parser.add_argument("--deadlock-iters", type=int, default=MAX_DEADLOCK_ITERATIONS, help="")
    parser.add_argument("--explore-iters", type=int, default=MAX_EXPLORATION_ITERATIONS, help="")
    parser.add_argument("--fullcost-iters", type=int, default=MAX_FULL_COST_ITERATIONS, help="")
    parser.add_argument("--enable-deadlock", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--enable-exploration", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--enable-full-cost", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--use-tensorboard", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    parser.add_argument("--chtc", type=bool, default=False, action=argparse.BooleanOptionalAction, help="")
    args = parser.parse_args()

    run(args)