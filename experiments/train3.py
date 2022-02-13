import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import sys, os

sys.path.append("/home/damien/Documents/maddpg/")

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tf_slim as slim


def parse_args():
    parser = argparse.ArgumentParser(
        "Reinforcement Learning experiments for multiagent environments"
    )
    # Environment
    parser.add_argument(
        "--scenario", type=str, default="simple", help="name of the scenario script"
    )
    parser.add_argument(
        "--max-episode-len", type=int, default=25, help="maximum episode length"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=60000, help="number of episodes"
    )
    parser.add_argument(
        "--num-adversaries", type=int, default=0, help="number of adversaries"
    )
    parser.add_argument(
        "--good-policy", type=str, default="maddpg", help="policy for good agents"
    )
    parser.add_argument(
        "--adv-policy", type=str, default="maddpg", help="policy of adversaries"
    )
    # Core training parameters
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="learning rate for Adam optimizer"
    )
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="number of episodes to optimize at the same time",
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
        "--exp-name", type=str, default="myExperiment", help="name of the experiment"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./policy/",
        help="directory in which training state and model should be saved",
    )
    parser.add_argument(
        "--save-rate",
        type=int,
        default=1000,
        help="save model once every time this many episodes are completed",
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        default="",
        help="directory in which training state and model are loaded",
    )
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=100000,
        help="number of iterations run for benchmarking",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./benchmark_files/",
        help="directory where benchmark data is saved",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="./learning_curves/",
        help="directory where plot data is saved",
    )
    parser.add_argument(
        "--use-same-good-agents",
        action="store_true",
        default=False,
        help="whether to use fixed good agent policy",
    )
    parser.add_argument(
        "--benchmark-run", type=int, default=1, help="affects benchmark file naming"
    )
    parser.add_argument(
        "--benchmark-filecount", type=int, default=20, help="number of files each run"
    )
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = slim.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = slim.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = slim.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
        )
    else:
        env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation
        )
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    # adversary agents
    for i in range(num_adversaries):
        trainers.append(
            trainer(
                "agent_%d" % i,
                # custom_mlp_model,
                model,
                obs_shape_n,
                env.action_space,
                i,
                arglist,
                local_q_func=(arglist.adv_policy == "ddpg"),
                num_units=None
                if arglist.num_units_adv == 64
                else arglist.num_units_adv,
            )
        )
    # good agents
    for i in range(num_adversaries, env.n):
        trainers.append(
            trainer(
                "agent_%d" % i,
                model,
                obs_shape_n,
                env.action_space,
                i,
                arglist,
                local_q_func=(arglist.good_policy == "ddpg"),
                num_units=None
                if arglist.num_units_good == 64
                else arglist.num_units_good,
            )
        )
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print(
            "Using good policy {} and adv policy {}".format(
                arglist.good_policy, arglist.adv_policy
            )
        )

        advAgentsNames = ["agent_%d" % i for i in range(num_adversaries)]
        goodAgentsNames = ["agent_%d" % i for i in range(num_adversaries, env.n)]

        # Initialize
        U.initialize()

        ### to use policies from different files
        if arglist.use_same_good_agents:
            goodAgentVars = []
            advAgentVars = []
            for _name in goodAgentsNames:
                goodAgentVars += tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=_name
                )
            for _name in advAgentsNames:
                advAgentVars += tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=_name
                )
            ###
            # goodAgentPolicy = "./policy-tag_s_base-40000-sameGA/"
            goodAgentPolicy = "./policy-tag_s_base-60000/"

            saver = tf.compat.v1.train.Saver(goodAgentVars)
            U.load_state(goodAgentPolicy, saver=saver)

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print("Loading previous state...")
            if arglist.use_same_good_agents:
                saver = tf.compat.v1.train.Saver(advAgentVars)
                U.load_state(arglist.load_dir, saver=saver)
            else:
                U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.compat.v1.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        captures = 0
        captures_timesteps = []
        benchmark_count = 0
        text = ""
        num_files_written = 0
        episode_objects = []

        from episode import Episode

        print("Starting iterations...")
        import datetime

        print(datetime.datetime.now())
        while True:
            # print('')
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = episode_step >= arglist.max_episode_len
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(
                    obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal
                )
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                # for i, info in enumerate(info_n):
                #     agent_info[-1][i].append(info_n['n'])
                # if train_step > arglist.benchmark_iters and (done or terminal):
                #     file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                #     print('Finished benchmarking, now saving...')
                #     with open(file_name, 'wb') as fp:
                #         pickle.dump(agent_info[:-1], fp)
                #     break
                def is_collision(agent1, agent2):
                    delta_pos = agent1.state.p_pos - agent2.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dist_min = agent1.size + agent2.size
                    return True if dist < dist_min else False

                for agent in env.agents:
                    if not agent.adversary:
                        for a in env.agents:
                            if a.adversary:
                                if is_collision(agent, a):
                                    captures += 1
                                    captures_timesteps.append(episode_step)
                if done or terminal:
                    # end of each episode
                    ep = Episode(captures, captures_timesteps)
                    episode_objects.append(ep)
                    captures = 0
                    captures_timesteps = []
                    benchmark_count += 1
                    num = 10000  # save every num episodes
                    if benchmark_count % num == 0:
                        file_name = (
                            arglist.benchmark_dir
                            + arglist.exp_name
                            + "-"
                            + str(benchmark_count - num)
                            + "_"
                            + str(benchmark_count)
                            + "_"
                            + str(arglist.benchmark_run)
                            + ".pkl"
                        )
                        with open(file_name, "wb") as fp:
                            pickle.dump(episode_objects, fp)
                        episode_objects = []
                        num_files_written += 1
                        print(
                            f"{num} iterations recorded and benchmarked, {num_files_written} files total"
                        )
                    if num_files_written == arglist.benchmark_filecount:
                        break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                if arglist.use_same_good_agents:
                    if agent.name in advAgentsNames:
                        agent.preupdate()
                    else:
                        # do not update good agent policy
                        continue
                else:
                    agent.preupdate()
            for agent in trainers:
                if arglist.use_same_good_agents:
                    if agent.name in advAgentsNames:
                        loss = agent.update(trainers, train_step)
                    else:
                        # do not update good agent policy
                        continue
                else:
                    loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print(
                        "steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                            train_step,
                            len(episode_rewards),
                            np.mean(episode_rewards[-arglist.save_rate :]),
                            round(time.time() - t_start, 3),
                        )
                    )
                else:
                    print(
                        "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                            train_step,
                            len(episode_rewards),
                            np.mean(episode_rewards[-arglist.save_rate :]),
                            [
                                np.mean(rew[-arglist.save_rate :])
                                for rew in agent_rewards
                            ],
                            round(time.time() - t_start, 3),
                        )
                    )
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate :]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate :]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + "_rewards.pkl"
                with open(rew_file_name, "wb") as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = (
                    arglist.plots_dir + arglist.exp_name + "_agrewards.pkl"
                )
                with open(agrew_file_name, "wb") as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print("...Finished total of {} episodes.".format(len(episode_rewards)))
                break
        print(datetime.datetime.now())


if __name__ == "__main__":
    arglist = parse_args()
    train(arglist)
