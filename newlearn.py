import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
import os
from petrienv import PetriEnv
from deadlockenv import DeadlockEnv
import time
import json
from datetime import datetime
from constants import *

# Rectify the numpy versions
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

start = datetime.now()

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

models_dir = f"models/Deadlock-PPO/"
logdir = f"logs/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# with open('test_run.json', encoding='utf-8') as fh:
with open(FILENAME, encoding='utf-8') as fh:
    json_obj = json.load(fh)

firstEnv = DeadlockEnv(json_obj)
secondEnv = PetriEnv(json_obj)

firstEnv.reset(0, {})
secondEnv.reset(0, {})
firstEnv = ActionMasker(firstEnv, mask_fn)  # Wrap to enable masking
secondEnv = ActionMasker(secondEnv, mask_fn)  # Wrap to enable masking

# Train on the deadlock environment first
model = MaskablePPO(MaskableActorCriticPolicy, firstEnv, verbose=1, tensorboard_log=logdir, device="auto")

if DEADLOCK_TRAINING:
    iters = 0
    while iters < MAX_DEADLOCK_ITERATIONS:
        iters += 1
        model.learn(total_timesteps=DEADLOCK_TIMESTEPS)
        if iters % DEADLOCK_ITERATION_SAVE_INTERVAL == 0:
            model.save(f"{models_dir}/Deadlock-PPO-deadlock-{iters}")
    # After training, save and load the model to change environments for the next round of training
    model.save(f"{models_dir}/Deadlock-PPO-deadlock-final")
    model = MaskablePPO.load(f"{models_dir}/Deadlock-PPO-deadlock-final", secondEnv, verbose=1, tensorboard_log=logdir, device="auto")
else:
    model = MaskablePPO(MaskableActorCriticPolicy, secondEnv, verbose=1, tensorboard_log=logdir, device="auto")
    # model = PPO('MlpPolicy', secondEnv, verbose=1, tensorboard_log=logdir)

if PPO_TRANING:
    # Train on the actual environment after we've learned to avoid deadlock scenarios
    iters = 0
    while iters < MAX_PPO_ITERATIONS:
        iters += 1
        model.learn(total_timesteps=PPO_TIMESTEPS)
        if iters % PPO_ITERATION_SAVE_INTERVAL == 0:
            model.save(f"{models_dir}/Deadlock-PPO-{iters}")

print("TOTAL TIME: {0}".format(datetime.now() - start))