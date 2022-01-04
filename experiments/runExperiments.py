from pickle import load
import subprocess as sp
import os

cmdBase = "python train.py".split()
cmdNumAdv = "--num-adversaries 3".split()
cmdNumEpisodes = "--num-episodes 60000".split()
cmdNumUnits = "--num-units 64".split()
cmdScenario = "--scenario tag_s_base".split()
cmdLoadDir = "--load-dir './policy-tag_s_base_60000/'".split()
cmdSaveDir = "--save-dir './policy-tag_s_base-60000/".split()
cmdExpName = "--exp-name tag_s_base".split()
cmdRestore = "--restore".split()
cmdBenchmark = "--benchmark".split()

allCmds = [
    cmdBase,
    cmdNumAdv,
    cmdNumEpisodes,
    cmdNumUnits,
    cmdScenario,
    cmdLoadDir,
    cmdSaveDir,
    cmdExpName,
    cmdRestore,
    cmdBenchmark,
]


def generateFullCommand(restore, benchmark):
    fullCmd = []
    for cmd in allCmds:
        if cmd == cmdLoadDir and not restore:
            continue
        if cmd == cmdRestore and not restore:
            continue
        if cmd == cmdBenchmark and not benchmark:
            continue
        fullCmd += cmd
    return list(map(lambda x: str(x), fullCmd))


# train 10,000 then benchmark? till 12,000?


if __name__ == "__main__":

    numIterations = 15

    numEpisodes = 10000
    numUnits = 64
    scenario = "tag_s_comm"

    initialDir = "./policy-tag_s_comm_LONG"
    initialExpName = "tag_s_comm_LONG"

    cmdNumEpisodes[1] = numEpisodes
    cmdNumUnits[1] = numUnits
    cmdScenario[1] = scenario

    with open(os.path.join("logs", f"{initialExpName}-log.txt"), "a") as f:
        loadDir = ""
        saveDir = ""
        fullCommand = []
        fullCommandBenchmark = []
        for i in range(1, numIterations + 1):
            expName = initialExpName + f"_{i*numEpisodes}"
            cmdExpName[1] = expName
            if i == 1:
                saveDir = initialDir + f"_{i*numEpisodes}/"
                cmdSaveDir[1] = saveDir
                fullCommand = generateFullCommand(restore=False, benchmark=False)
                fullCommandBenchmark = generateFullCommand(
                    restore=False, benchmark=True
                )
            else:
                loadDir = initialDir + f"_{(i-1)*numEpisodes}/"
                saveDir = initialDir + f"_{i*numEpisodes}/"
                cmdLoadDir[1] = loadDir
                cmdSaveDir[1] = saveDir
                fullCommand = generateFullCommand(restore=True, benchmark=False)
                fullCommandBenchmark = generateFullCommand(
                    restore=False, benchmark=True
                )

            print(fullCommand)
            print(fullCommandBenchmark)

            f.write(" ".join(fullCommand) + "\n")
            f.write(" ".join(fullCommandBenchmark) + "\n")

            sp.run(fullCommand, stdout=f, text=True)
            sp.run(fullCommandBenchmark, stdout=f, text=True)
