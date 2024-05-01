from stable_baselines3 import PPO
import os
from petrienv import PetriEnv
import time
import json

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# with open('test_run.json', encoding='utf-8') as fh:
with open('cost_net.json', encoding='utf-8') as fh:
    json_obj = json.load(fh)

env = PetriEnv(json_obj)

# print(env.num_places)
# print(env.num_transitions)
# print(env.goal_state)
# print(env.initial_marking)
# print(env.C)
# exit(1)

env.reset(0)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 20000
MAX_ITERS = 100
iters = 0
while iters < MAX_ITERS:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS)
	model.save(f"{models_dir}/PPO-{iters}")