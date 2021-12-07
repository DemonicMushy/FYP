from episode import Episode
import pickle
import os
import re
import pandas as pd


def parseFiles(dir):
    objs = {}
    for subdir, dirs, files in os.walk(dir):
        basenames = []
        for file in files:
            basename = re.split("-", file)[:-1]
            basename = '-'.join(basename)
            if basename not in basenames:
                basenames.append(basename)
        basenames.sort()
        for basename in basenames:
            episodes = []
            objs[basename] = {}
            for file in files:
                n = re.split("-", file)[:-1]
                n = '-'.join(n)
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
    count = 0
    for epi in obj["episodes"]:
        count += 1 if epi.first_capture_timestep != None else 0
    obj["first_capture_timestep"] = count


if __name__ == "__main__":
    fileDir = "./benchmark_files/"
    experiments = parseFiles(fileDir)
    df = pd.DataFrame(
        columns=["Scenario", "Total Episodes", "Num Captures", "Capture Rate"]
    )
    print(df)
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
    print(df)