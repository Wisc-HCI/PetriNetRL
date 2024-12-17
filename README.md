# PetriNetRL
## Description
A reinforcement learning approach for petri nets regarding human-robot collaboration tasks. Still a WIP.

## Non-Docker Setup
You'll need to download and set up [Stable Baselines3 - Contrib](https://sb3-contrib.readthedocs.io/en/master/), which implements the [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/guide/examples.html#maskableppo) implementation utilized in this repo. All Python requirements are outlined in `requirements.txt`, meaning you can run `pip install -r requirements.txt` to install all needed Python packages.

If you want to view the training logs, you can also set up a docker image for [Tensorboard](https://hub.docker.com/r/volnet/tensorflow-tensorboard). 

## Docker Setup
Download [this docker image](https://hub.docker.com/repository/docker/nwhite365/maskableppo/general) which is a fully configured environment (minus tensorboard). If you want to view the training logs, you can also set up an additional docker image for [Tensorboard](https://hub.docker.com/r/volnet/tensorflow-tensorboard). 

After cloning the image, navigate to where this repo is installed and run: `docker run --rm=true -it -v ${pwd}:/scratch -w /scratch <docker image> /bin/bash`. This runs the docker image and connects a volume containing the current directory to it at `/scratch`. This will allow you to run the various python scripts in the environment.

## Running
### Model Building
To train a model, you can run `python build_model.py` with the following parameters:
- `--input-file <path_to_file: string>`: Default is __None__. This file provides the metadata for the job.
- `--base-model <path_to_file: string>`: Default is __None__. The use of this parameters specifies a model to load and start the learning process from.
- `--prepend <name: string>`: Default is __None__. This is an additional string you can prepend to output file names.
- `--process <id: int>`: Default is __0__. This is an integer appended to the output model filename for use in parallel setups.
- `--iters <num: int>`: Default is __50__.
- `--use-tensorboard`: Default is __False__. Addition of flag is __True__. This flag adjusts adjusts whether to output log files.
- `--chtc`: Default is __False__. Addition of flag is __True__. This flag adjusts the execution for use on the center for high throughput computing.

At minimum, you need to specify an input file. These input files are __job.json__ files produced by the [Allocobot](https://github.com/Wisc-HCI/allocobot) repo. Job files contain all information regarding the relative weighting of different metrics; the tasks, primitives, environmental targets, and agents involved in a job; and the Petri Net with all the costs associated for transitions. An example job.json file has been included in this repo, and represents a mock task.

### Model Sampling and Understanding
You can run `python evaluate_model.py` to sample allocations and timelines from your learned model, with the following parameters:
- `--input-file <path_to_file: string>`: Default is __None__. This file provides the metadata for the job.
- `--model <path_to_file: string>`: Default is __None__. The use of this parameters specifies a model to load and sample from.
- `--output <filename: string>`: Default is __None__. If __None__, the output filename will be the same as the input-file name with `.json` replaced with `-output.csv`. 
- `--n-steps <num: int>`: Default is __1000__. This represents the maximum number of steps to run the model for. If the model reaches the goal state, it will terminate before `n_steps` is reached.
- `--n-samples <num: int>`: Default is __0__. This represents the number of sample traces to compare from the model.

At minimum, an input file and model is required for the evaluation to work. This leaves the model in a stocastic mode, allowing you to sample multiple traces.

### Allocation and Misc Understanding
You can run `python create_allocation_visual.py` with the following parameters to understand some of the metrics from the saved trace from the model:
- `--input-file <path_to_file: string>`: Default is __None__. This file provides the metadata for the job.
- `--csv-file <path_to_file: string>`: Default is __None__. This file represents the output from the `evaluate_model` script.

Running this script will analyze the allocation of tasks from the job between the various agents, and output metrics such as task time. Additional analysis will require investigating the CSV outputs manually.

If you want to combine this with the prior sampling, you can use the `eval.sh` script to run both commands. Eval works as follows: `eval.sh <input-file> <n-steps> <n-samples> <model>`, and will first run `evaluate_model` followed by `create_allocation_visual`.