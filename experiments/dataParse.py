from episode import Episode
import pickle
import os


def parseFiles(dir, filename):
    objs = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if filename in file:
                with open(dir + file, "rb") as fp:
                    obj = pickle.load(fp)
                    objs += obj
    return objs


def getTotalEpisodesWithCaptures(episodes):
    count = 0
    for epi in episodes:
        count += 1 if epi.num_captures > 0 else 0
    return count


def getAverageFirstCaptureTimestep(episodes):
    count = 0
    for epi in episodes:
        count += 1 if epi.first_capture_timestep != None else 0
    return count


if __name__ == "__main__":
    fileDir = "./benchmark_files/"
    filename = "simple_tag"
    episodes = parseFiles(fileDir, filename)
    totalEpisodes = len(episodes)
    totalEpisodesWithCaptures = getTotalEpisodesWithCaptures(episodes)

    print("Total Episodes: ", totalEpisodes)
    print("Total Episodes with captures: ", totalEpisodesWithCaptures)
