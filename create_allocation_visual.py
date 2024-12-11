from constants import *
from utils import *
import csv
import argparse
import pandas as pd

def task_completion_time(csv_file):
    time = 0.0
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        count = 0
        for row in reader:
            if count == 0:
                count += 1
                continue
            if float(row[5]) > time:
                time = float(row[5])
            

    return time

def run(arguments):
    # Determine which input json file to use 
    f = FILENAME # default "cost_net.json"
    if arguments.input_file is not None:
        f = arguments.input_file
    
    # Load petrinet data from json (transitions, places)
    [_json_obj, _weights, json_task, _targets_obj, _primitives_obj, json_agents] = LOAD_JOB_FILE(f)

    task_steps = [str(i) for i in range(len(list(json_task.keys())))]
    tasks_ordered = [None for _ in json_task.keys()]
    task_assignment = [None for _ in json_task.keys()]
    for key in json_task.keys():
        if tasks_ordered[json_task[key]["order"]-1] is None:
            tasks_ordered[json_task[key]["order"]-1] = [json_task[key]["name"]]
        else:
            tasks_ordered[json_task[key]["order"]-1].append(json_task[key]["name"])

    with open(arguments.csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if "decide" in row[0]:
                task_name =  row[0].split(" decide")[0].strip()
                idx = -1
                for eidx, i in enumerate(tasks_ordered):
                    if i is not None and task_name in i:
                        idx = eidx
                task_assignment[idx] = row[3]


    data = pd.DataFrame(dict(zip(task_steps, task_assignment)),
                        index=[0])
    print(data)
    print("Full task time (s): {}".format(task_completion_time(arguments.csv_file)))
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=None, help="")
    parser.add_argument("--csv-file", type=str, default=None, help="")
    args = parser.parse_args()

    run(args)