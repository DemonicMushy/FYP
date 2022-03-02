from multiprocessing import Process
import argparse

cmdBase = "python runExperiments.py".split()
cmdScenario = "--scenario tag_s_base".split()
cmdStartIter = "--start-iter 1".split()
cmdEndIter = "--end-iter 12".split()
cmdNumEpisodes = "--num-episodes 60000".split()
cmdNumUnits = "--num-units 64".split()
cmdNumUnitsAdv = "--num-units-adv 64".split()
cmdNumUnitsGood = "--num-units-good 64".split()
cmdInitialExpName = "--initial-exp-name myExperiment".split()
cmdInitialDir = "--initial-dir './policy/".split()
cmdBenchmark = "--benchmark yes".split()
cmdBenchmarkInterval = "--benchmark-interval 1".split()
cmdBenchmarkRun = "--benchmark-run 1".split()
cmdBenchmarkFilecount = "--benchmark-filecount 1".split()

allCmds = [
    # cmdBase,
    cmdStartIter,
    cmdEndIter,
    cmdNumEpisodes,
    cmdNumUnits,
    cmdNumUnitsAdv,
    cmdNumUnitsGood,
    cmdScenario,
    cmdInitialExpName,
    cmdInitialDir,
    cmdBenchmark,
    cmdBenchmarkInterval,
    cmdBenchmarkRun,
    cmdBenchmarkFilecount,
]


def parse_args():
    parser = argparse.ArgumentParser(
        "Custom training script to call runExperiments(''|2|3).py multiple times"
    )
    parser.add_argument(
        "--file", type=str, default="runExperiments.py", help="name of the python file"
    )
    parser.add_argument(
        "--file-runs", type=int, default=2, help="number of times to run"
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


def generateFullCommand():
    fullCmd = []
    for cmd in allCmds:
        fullCmd += cmd
    return list(map(lambda x: str(x), fullCmd))


if __name__ == "__main__":
    arglist = parse_args()
    # print(arglist)

    initialDir = arglist.initial_dir
    initialExpName = arglist.initial_exp_name

    cmdBase[1] = arglist.file
    cmdScenario[1] = arglist.scenario
    cmdStartIter[1] = arglist.start_iter
    cmdEndIter[1] = arglist.end_iter
    cmdNumEpisodes[1] = arglist.num_episodes
    cmdNumUnits[1] = arglist.num_units
    cmdNumUnitsAdv[1] = arglist.num_units_adv
    cmdNumUnitsGood[1] = arglist.num_units_good
    cmdBenchmark[1] = arglist.benchmark
    cmdBenchmarkInterval[1] = arglist.benchmark_interval
    cmdBenchmarkRun[1] = arglist.benchmark_run
    cmdBenchmarkFilecount[1] = arglist.benchmark_filecount

    if arglist.file == "runExperiments.py":
        from runExperiments import runExp
        from runExperiments import parse_args as parse_args_other
    elif arglist.file == "runExperiments2.py":
        from runExperiments2 import runExp
        from runExperiments2 import parse_args as parse_args_other
    elif arglist.file == "runExperiments3.py":
        from runExperiments3 import runExp
        from runExperiments3 import parse_args as parse_args_other
    else:
        raise Exception("Invalid file argument")

    processes = []
    for i in range(arglist.file_runs):
        directory = initialDir + "_" + str(i)
        experimentName = initialExpName + "_" + str(i)
        cmdInitialDir[1] = directory
        cmdInitialExpName[1] = experimentName
        # print(" ".join(generateFullCommand()))
        arglist_other = parse_args_other(generateFullCommand())
        # print(arglist_other)
        p = Process(target=runExp, args=(arglist_other,))
        p.start()
        processes.append(p)

    for p in processes:
        # p.wait()
        p.join()
