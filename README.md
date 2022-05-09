[GitHub Link to Repo](https://github.com/DemonicMushy/FYP)


This code is based off [OpenAI's MADDPG implementation](https://github.com/openai/maddpg/). A copy of its README can be found [below](#multi-agent-deep-deterministic-policy-gradient-maddpg).

## Dependencies and Versions

- Experiemnts were conducted with these: Python (3.9.7), OpenAI gym (0.21.0), tensorflow (2.6.0), tf-slim (1.1.0), numpy (1.19.5)

## Tensorflow v1 to v2

The original repo was written for Tensorflow v1. Hence I had to follow [this process](https://www.tensorflow.org/guide/migrate) to migrate to v2, and hence the need for `tf-slim` package.


## Examples
Be sure that your directory is at `experiments` before running the example commands
```
cd experiments
```

Example to train
```
python train3.py --num-adversaries 3 --num-episodes 10000 --num-units 64 --scenario tag_s_comm --save-dir ./policy-tag_s_comm_10000/
```

Example to render simulation of trained policy
```
python train3.py --num-adversaries 3 --num-episodes 10000 --num-units 64 --scenario tag_s_comm --load-dir ./policy-tag_s_comm_10000/ --display
```

Example to run benchmark
```
python train3.py --num-adversaries 3 --num-episodes 10000 --num-units 64 --scenario tag_s_comm --load-dir ./policy-tag_s_comm_10000/ --exp-name tag_s_comm_10000 --benchmark
```

## Difference between train, train2, train3 (`./experiments/`)

`train3.py` is using the original 2 layer ReLU MLP. Do reference to this to compare the difference between the others

`train2.py` and `train.py` is where I manually change the number of layers and number of units of each layer (hence arguments --num-units and such is rendered useless for these two)
- see function `custom_mlp_model` and related where the neural network configuration is defined
- see function `get_trainers` where it is used


## Important files to look at

- `./multiagent/environment.py`
  - Find `# zxcv` to refer to area where I added code, everything else should be from the original code (do compare it yourself to be sure)
  - This is where the calculations for the adversaries' communicated values happen (!)
- `./multiagent/core.py`
  - To understand more about the entities in the environments
  - No changes to code from original 

<br>

## What are `runExperiments?.py` and `runMultipleExps.py` for?

*Suggest that you understaind `train.py` and scenarios first before caring about these files.*

`runExperiments?.py`: These files are scripts to basically run the `train?.py` scripts sequentially so that I can train for 2,000 episodes, then evaluate, then train another 2,000, then evaluate again, and so on.
- It is a very rough and dirty script that I made, so you should definitely look through and understand it first if you want to use it
- Do take note of the `parse_args` and `parse_args_other` and notice that they may be different

`runMultipleExps.py`: This file is a script to run multiple `runExperiments?.py` scripts.
- Once again it is rough and dirty script, so use with caution
- Do take note of the `parse_args` and `parse_args_other` and notice that they may be different

<br>

## Scenarios explained (`./multiagent/scenarios/`)

Do compare the differences in the lines of codes between scenarios to understand where there are changes (mainly found in the adversary observation space function, and make_world function)

`diff <filename> <filename2>` is a good way to view the file differences.

#### simple*.py
The original scenarios from the MPE repository.

#### tag_s_base.py
The base scenario from `simple_tag.py`

#### tag_s_base_wDistance.py
The Distance scenario mentioned in the report.

#### tag_s_comm.py
The Communication scenario mentioned in the report.

#### tag_s_los_*.py
Copies of the above scenarios but with Line-Of-Sight (LOS) blocking. (UNUSED AND NOT REFERENCE IN REPORT)

#### tag_s_lying1.py
The Communication scenario with lying enabled (all adversary can lie).

#### tag_s_lying1single.py
The Communication scenario with lying enabled (single liar).


<br>

## Evaluation Related

The script `dataPase.py` was made to parse through the individual `*.pkl` files generated during the benchmarking step of `train?.py` and output a csv file with the compiled results.


<br>

## `commands.txt`

This text file contains most of the command line entries I used in the process of my FYP. The earlier in the file, the less likely to make any sense, so don't fret if it doesn't make sense. 

Do only use the most latest entries as a reference to the command line arguments I passed to the various scripts.

<br>

## Do take note:

"--use-same-good-agents" argument
- This option was suppose to allow evaluation of different adversary agent policies against the same policy for good agents
- The good agent policy file location was suppose to be changed inside `train?.py` itself
- This option has not been touched or used for some time so I cannot guarentee its functionality

Some defaults
- Training episodes: every 2,000 until 60,000

---
The section below is the README of the [original maddpg repo](https://github.com/openai/maddpg/) excluding the citation portion.

---

# Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

This is the code for implementing the MADDPG algorithm presented in the paper:
[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).
It is configured to be run in conjunction with environments from the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).
Note: this codebase has been restructured since the original paper, and the results may
vary from those reported in the paper.

**Update:** the original implementation for policy ensemble and policy estimation can be found [here](https://www.dropbox.com/s/jlc6dtxo580lpl2/maddpg_ensemble_and_approx_code.zip?dl=0). The code is provided as-is. 

## Installation

- To install, `cd` into the root directory and type `pip install -e .`

- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), tensorflow (1.8.0), numpy (1.14.5)

## Case study: Multi-Agent Particle Environments

We demonstrate here how the code can be used in conjunction with the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

- Download and install the MPE code [here](https://github.com/openai/multiagent-particle-envs)
by following the `README`.

- Ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH` (e.g. in `~/.bashrc` or `~/.bash_profile`).

- To run the code, `cd` into the `experiments` directory and run `train.py`:

``python train.py --scenario simple``

- You can replace `simple` with any environment in the MPE you'd like to run.

## Command-line options

### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"simple"`)

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `60000`)

- `--num-adversaries`: number of adversaries in the environment (default: `0`)

- `--good-policy`: algorithm used for the 'good' (non adversary) policies in the environment
(default: `"maddpg"`; options: {`"maddpg"`, `"ddpg"`})

- `--adv-policy`: algorithm used for the adversary policies in the environment
(default: `"maddpg"`; options: {`"maddpg"`, `"ddpg"`})

### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

### Checkpointing

- `--exp-name`: name of the experiment, used as the file name to save all results (default: `None`)

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/tmp/policy/"`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--load-dir`: directory where training state and model are loaded from (default: `""`)

### Evaluation

- `--restore`: restores previous training state stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), and continues training (default: `False`)

- `--display`: displays to the screen the trained policy stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), but does not continue training (default: `False`)

- `--benchmark`: runs benchmarking evaluations on saved policy, saves results to `benchmark-dir` folder (default: `False`)

- `--benchmark-iters`: number of iterations to run benchmarking for (default: `100000`)

- `--benchmark-dir`: directory where benchmarking data is saved (default: `"./benchmark_files/"`)

- `--plots-dir`: directory where training curves are saved (default: `"./learning_curves/"`)

## Code structure

- `./experiments/train.py`: contains code for training MADDPG on the MPE

- `./maddpg/trainer/maddpg.py`: core code for the MADDPG algorithm

- `./maddpg/trainer/replay_buffer.py`: replay buffer code for MADDPG

- `./maddpg/common/distributions.py`: useful distributions used in `maddpg.py`

- `./maddpg/common/tf_util.py`: useful tensorflow functions used in `maddpg.py`


