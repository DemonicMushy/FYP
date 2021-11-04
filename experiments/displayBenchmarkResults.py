import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Parser for benchmark files")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--benchmark-file", type=str, default="", help="benchmark file name")
    return parser.parse_args()

def processFile(filename, filedir):
    fullfilename = filedir + filename
    with open(fullfilename, 'rb') as fp:
        file = pickle.load(fp)
    print(file)


if __name__ == '__main__':
    args = parse_args()
    processFile(args.benchmark_file, args.benchmark_dir)