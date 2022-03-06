import os
import argparse
from multiprocessing import Process
from datetime import datetime
from train3 import train
from train3 import parse_args as parse_args_other

cmdBase = "python train3.py".split()
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
cmdBenchmarkFilecount = "--benchmark-filecount 1".split()

allCmds = [
    # cmdBase,
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


def parse_args(cmd=None):
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
        "--benchmark-filecount", type=int, default=1, help="number of files each run"
    )
    return parser.parse_args(cmd)


def runExp(arglist):
    benchmark = arglist.benchmark

    startingIteration = arglist.start_iter
    endingIteration = arglist.end_iter

    initialDir = arglist.initial_dir
    initialExpName = arglist.initial_exp_name

    cmdNumEpisodes[1] = arglist.num_episodes
    cmdNumUnits[1] = arglist.num_units
    cmdNumUnitsAdv[1] = arglist.num_units_adv
    cmdNumUnitsGood[1] = arglist.num_units_good
    cmdScenario[1] = arglist.scenario
    cmdBenchmarkRun[1] = arglist.benchmark_run
    cmdBenchmarkFilecount[1] = arglist.benchmark_filecount

    benchmark_interval = arglist.benchmark_interval
    benchmark_index = 0

    loadDir = ""
    saveDir = ""
    fullCommand = []
    fullCommandBenchmark = []
    for i in range(startingIteration, endingIteration + 1):
        with open(os.path.join("logs", f"{initialExpName}-log.txt"), "a") as f:
            expName = initialExpName + f"_{i*arglist.num_episodes}"
            cmdExpName[1] = expName
            if i == 1:
                saveDir = initialDir + f"_{i*arglist.num_episodes}/"
                cmdSaveDir[1] = saveDir
                fullCommand = generateFullCommand(restore=False, benchmark=False)
                fullCommandBenchmark = generateFullCommand(
                    restore=False, benchmark=True
                )
                arglist_other = parse_args_other(fullCommand)
                arglist_other_benchmark = parse_args_other(fullCommandBenchmark)
            else:
                loadDir = initialDir + f"_{(i-1)*arglist.num_episodes}/"
                saveDir = initialDir + f"_{i*arglist.num_episodes}/"
                cmdLoadDir[1] = loadDir
                cmdSaveDir[1] = saveDir
                fullCommand = generateFullCommand(restore=True, benchmark=False)
                fullCommandBenchmark = generateFullCommand(
                    restore=False, benchmark=True
                )
                arglist_other = parse_args_other(fullCommand)
                arglist_other_benchmark = parse_args_other(fullCommandBenchmark)

            if benchmark == "only":
                pass
            else:
                trainStart = datetime.now()
                print("Train start:", trainStart, file=f)
                print(" ".join(fullCommand), file=f)
                # f.write(" ".join(fullCommand) + "\n")
                # sp.run(fullCommand, stdout=f, text=True)
                print(arglist_other)
                p = Process(target=train, args=(arglist_other,))
                p.start()
                p.join()
                trainEnd = datetime.now()
                print("Train End:", trainEnd, file=f)
                print("Train Duration:", trainEnd - trainStart, file=f)

            if benchmark == "no":
                # redundant since checking for yes and only below
                continue

            if (benchmark == "yes") or (benchmark == "only"):
                benchmark_index += 1
                if benchmark_index % benchmark_interval == 0:
                    # benchmark every set interval (eg 2, )
                    benchStart = datetime.now()
                    print("Bench start:", benchStart, file=f)
                    print(" ".join(fullCommandBenchmark), file=f)
                    # f.write(" ".join(fullCommandBenchmark) + "\n")
                    # sp.run(fullCommandBenchmark, stdout=f, text=True)
                    print(arglist_other_benchmark)
                    p = Process(target=train, args=(arglist_other_benchmark,))
                    p.start()
                    p.join()
                    benchEnd = datetime.now()
                    print("Bench End:", benchEnd, file=f)
                    print("Bench Duration:", benchEnd - benchStart, file=f)


if __name__ == "__main__":
    arglist = parse_args()
    runExp(arglist)
