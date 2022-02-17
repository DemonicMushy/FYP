from episode import Episode
import pickle
import os
import re
import pandas as pd
import numpy as np


def parseFiles(dir):
    objs = {}
    for subdir, dirs, files in os.walk(dir):
        basenames = []
        for file in files:
            if file == ".DS_Store":
                continue
            basename = re.split("-", file)[:-1]
            basename = "-".join(basename)
            if basename not in basenames:
                basenames.append(basename)
        basenames.sort()
        for basename in basenames:
            episodes = []
            objs[basename] = {}
            for file in files:
                n = re.split("-", file)[:-1]
                n = "-".join(n)
                if basename == n:
                    with open(dir + file, "rb") as fp:
                        obj = pickle.load(fp)
                        episodes += obj
            objs[basename]["episodes"] = episodes
            objs[basename]["total_episodes"] = len(episodes)
    return objs


def getTotalEpisodesWithCaptures(obj):
    count = 0
    for epi in obj["episodes"]:
        count += 1 if epi.num_captures > 0 else 0
    obj["num_captures"] = count


def getAverageFirstCaptureTimestep(obj):
    timesteps = []
    for epi in obj["episodes"]:
        if epi.first_capture_timestep != None:
            timesteps.append(epi.first_capture_timestep)
    obj["ave_first_capture_timestep"] = np.mean(timesteps)


def getStd(obj, bucketSize):
    a = []
    count = 0
    counter = 0
    for epi in obj["episodes"]:
        count += 1 if epi.num_captures > 0 else 0
        counter += 1
        if counter % bucketSize == 0:
            a.append(count)
            counter = 0
            count = 0
    obj["std"] = np.std(a)


def getMean(obj, bucketSize):
    a = []
    count = 0
    counter = 0
    for epi in obj["episodes"]:
        count += 1 if epi.num_captures > 0 else 0
        counter += 1
        if counter % bucketSize == 0:
            a.append(count)
            counter = 0
            count = 0
    obj["mean"] = np.mean(a)


if __name__ == "__main__":
    fileDir = "./benchmark_files/"
    experiments = parseFiles(fileDir)
    bucketSize = 1000
    df = pd.DataFrame(
        columns=[
            "Scenario",
            "Total Episodes",
            # "Num Captures",
            "Capture Rate",
            # f"Mean (Per {bucketSize})",
            f"Std (Per {bucketSize})",
        ]
    )

    for exp in experiments:
        getTotalEpisodesWithCaptures(experiments[exp])
        getAverageFirstCaptureTimestep(experiments[exp])
        getStd(experiments[exp], bucketSize)
        getMean(experiments[exp], bucketSize)

    for exp in experiments:

        dict1 = {
            "Scenario": exp,
            "Total Episodes": experiments[exp]["total_episodes"],
            # "Num Captures": experiments[exp]["num_captures"],
            "Capture Rate": "{:.2%}".format(
                experiments[exp]["num_captures"] / experiments[exp]["total_episodes"]
            ),
            # f"Mean (Per {bucketSize})": experiments[exp]["mean"],
            f"Std (Per {bucketSize})": "{:.2f}".format(experiments[exp]["std"]),
        }
        df = df.append(dict1, ignore_index=True)
    pd.set_option("display.max_rows", None)
    df.set_index("Scenario", inplace=True)

    df.to_csv("benchmark.csv")
    # print(df)
