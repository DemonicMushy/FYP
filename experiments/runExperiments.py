from pickle import load
import subprocess as sp
import os
import argparse

cmdBase = "python train.py".split()
cmdNumAdv = "--num-adversaries 3".split()
cmdNumEpisodes = "--num-episodes 60000".split()
cmdNumUnits = "--num-units 64".split()
cmdNumUnitsAdv = "--num-units-adv 64".split()
cmdNumUnitsGood = "--num-units-good 64".split()
cmdScenario = "--scenario tag_s_base".split()
cmdLoadDir = "--load-dir './policy-tag_s_base_60000/'".split()
cmdSaveDir = "--save-dir './policy-tag_s_base-60000/".split()
cmdExpName = "--exp-name tag_s_base".split()
cmdRestore = "--restore".split()
cmdBenchmark = "--benchmark".split()
cmdBenchmarkRun = "--benchmark-run 1".split()
cmdBenchmarkFilecount = "--benchmark-filecount 20".split()

allCmds = [
    cmdBase,
    cmdNumAdv,
    cmdNumEpisodes,
    cmdNumUnits,
    cmdNumUnitsAdv,
    cmdNumUnitsGood,
    cmdScenario,
    cmdLoadDir,
    cmdSaveDir,
    cmdExpName,
    cmdRestore,
    cmdBenchmark,
    cmdBenchmarkRun,
    cmdBenchmarkFilecount,
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


def parse_args():
    parser = argparse.ArgumentParser(
        "Custom training script to call train.py multiple times and benchmark in between"
    )
    # Environment
    parser.add_argument(
        "--scenario", type=str, default="tag_s_base", help="name of the scenario script"
    )
    parser.add_argument("--start-iter", type=int, default=1, help="starting iter num")
    parser.add_argument("--end-iter", type=int, default=12, help="ending iter num")
    # Core training parameters
    parser.add_argument(
        "--num-episodes", type=int, default=60000, help="number of episodes"
    )
    parser.add_argument(
        "--num-units", type=int, default=64, help="number of units in the mlp"
    )
    parser.add_argument(
        "--num-units-adv",
        type=int,
        default=64,
        help="number of units in the mlp for adv agents",
    )
    parser.add_argument(
        "--num-units-good",
        type=int,
        default=64,
        help="number of units in the mlp for good agents",
    )
    # Checkpointing
    parser.add_argument(
        "--initial-exp-name",
        type=str,
        default="myExperiment",
        help="name of the experiment",
    )
    parser.add_argument(
        "--initial-dir",
        type=str,
        default="./policy/",
        help="directory path",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="yes",
        help="whether to benchmark, yes/no/only",
    )
    parser.add_argument(
        "--benchmark-interval",
        type=int,
        default=1,
        help="benchmark interval",
    )
    parser.add_argument(
        "--benchmark-run", type=int, default=1, help="affects benchmark file naming"
    )
    parser.add_argument(
        "--benchmark-filecount", type=int, default=20, help="number of files each run"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arglist = parse_args()
    print(arglist)

    benchmark = arglist.benchmark

    # numIterations = 6
    # startingIteration = 4
    # endingIteration = 6
    startingIteration = arglist.start_iter
    endingIteration = arglist.end_iter

    numEpisodes = arglist.num_episodes
    numUnits = arglist.num_units
    numUnitsAdv = arglist.num_units_adv
    numUnitsGood = arglist.num_units_good
    # scenario = "tag_s_los_base_wDistance"
    scenario = arglist.scenario

    # initialDir = "./policy-tag_s_los_base_wDistance_LONG"
    # initialExpName = "tag_s_los_base_wDistance_LONG"
    initialDir = arglist.initial_dir
    initialExpName = arglist.initial_exp_name

    benchmarkRun = arglist.benchmark_run
    benchmarkFilecount = arglist.benchmark_filecount

    cmdNumEpisodes[1] = numEpisodes
    cmdNumUnits[1] = numUnits
    cmdNumUnitsAdv[1] = numUnitsAdv
    cmdNumUnitsGood[1] = numUnitsGood
    cmdScenario[1] = scenario
    cmdBenchmarkRun[1] = benchmarkRun
    cmdBenchmarkFilecount[1] = benchmarkFilecount

    benchmark_interval = arglist.benchmark_interval
    benchmark_index = 0

    with open(os.path.join("logs", f"{initialExpName}-log.txt"), "a") as f:
        loadDir = ""
        saveDir = ""
        fullCommand = []
        fullCommandBenchmark = []
        for i in range(startingIteration, endingIteration + 1):
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

            if benchmark == "only":
                pass
            else:
                print(fullCommand)
                f.write(" ".join(fullCommand) + "\n")
                sp.run(fullCommand, stdout=f, text=True)

            if benchmark == "no":
                # redundant since checking for yes and only below
                continue

            if (benchmark == "yes") or (benchmark == "only"):
                benchmark_index += 1
                if benchmark_index % benchmark_interval == 0:
                    # benchmark every set interval (eg 2, )
                    print(fullCommandBenchmark)
                    f.write(" ".join(fullCommandBenchmark) + "\n")
                    sp.run(fullCommandBenchmark, stdout=f, text=True)
