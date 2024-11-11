from constants import *
from utils import *
import argparse

def verifyJobFile(json_obj):

    passMVC = True
    maxMVC = 0
    maxTransition = ""
    for i, place in enumerate(json_obj["transitions"]):
        for data in json_obj["transitions"][place]["metaData"]:
            if data["type"] == "mVC" and data["value"][1] > maxMVC:
                maxMVC = data["value"][1]
                maxTransition = place
            if data["type"] == "mVC" and data["value"][1] > 1.5:
                passMVC = False
    print(maxMVC, maxTransition)
    assert passMVC


def getAllOneTimeCosts(json_obj):
    for i, place in enumerate(json_obj["transitions"]):
        for data in json_obj["transitions"][place]["cost"]:
            if data["frequency"] == "once":
                print(place, data["value"])

def run(arguments):
    f = FILENAME # default "cost_net.json"
    if arguments.inputfile is not None:
        f = arguments.inputfile

    # Load petrinet data from json (transitions, places)
    [json_obj, _weights, _json_task] = LOAD_JOB_FILE(f)

    verifyJobFile(json_obj)
    getAllOneTimeCosts(json_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", type=str, default=None, help="")
    args = parser.parse_args()

    run(args)