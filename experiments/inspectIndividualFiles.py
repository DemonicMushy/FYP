from episode import Episode
import pickle
import os
import re
import pandas as pd
import numpy as np


def parseFiles(dir, targetScenario):
    objs = {}
    for subdir, dirs, files in os.walk(dir):
        basenames = []
        # for file in files:
        #     basename = re.split("-", file)[:-1]
        #     basename = '-'.join(basename)
        #     if basename not in basenames:
        #         basenames.append(basename)
        # basenames.sort()

        episodes = []
        # objs[basename] = {}
        for file in files:
            n = re.split("-", file)[:-1]
            n = '-'.join(n)
            if targetScenario == n:
                objs[file] = {}
                with open(dir + file, "rb") as fp:
                    obj = pickle.load(fp)
                    episodes = obj
                objs[file]["episodes"] = episodes
                objs[file]["total_episodes"] = len(episodes)
    return objs


def getTotalEpisodesWithCaptures(obj):
    count = 0
    for epi in obj["episodes"]:
        count += 1 if epi.num_captures > 0 else 0
    obj["num_captures"] = count


def getAverageFirstCaptureTimestep(obj):
    count = 0
    for epi in obj["episodes"]:
        count += 1 if epi.first_capture_timestep != None else 0
    obj["first_capture_timestep"] = count

def getStd(experiments):
    a = []
    for exp in experiments:
        a.append(experiments[exp]["num_captures"])
    return np.std(a)

def getMean(experiments):
    a = []
    for exp in experiments:
        a.append(experiments[exp]["num_captures"])
    return np.mean(a)

if __name__ == "__main__":
    fileDir = "./benchmark_files/"
    targetScenario = "tag_s_base"
    experiments = parseFiles(fileDir, targetScenario)
    df = pd.DataFrame(
        columns=["Scenario", "Total Episodes", "Num Captures", "Capture Rate"]
    )

    for exp in experiments:
        getTotalEpisodesWithCaptures(experiments[exp])
        getAverageFirstCaptureTimestep(experiments[exp])


    for exp in experiments:
        dict1 = {
            "Scenario": exp,
            "Total Episodes": experiments[exp]["total_episodes"],
            "Num Captures": experiments[exp]["num_captures"],
            "Capture Rate": "{:.2%}".format(
                experiments[exp]["num_captures"] / experiments[exp]["total_episodes"]
            ),
        }
        df = df.append(dict1, ignore_index=True)

    std = getStd(experiments)
    mean = getMean(experiments)

    print(df)
    print("Standard Deviation: ", std)
    print("Mean: ", mean)
