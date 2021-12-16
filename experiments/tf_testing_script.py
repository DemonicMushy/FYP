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
    for i in range(num_adversaries):
        trainers.append(
            trainer(
                "agent_%d" % i,
                model,
                obs_shape_n,
                env.action_space,
                i,
                arglist,
                local_q_func=(arglist.adv_policy == "ddpg"),
            )
        )
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
            )
        )
    return trainers


class MyObject:
    def __init__(
        self,
        scenario,
        max_episode_len,
        num_episodes,
        num_adversaries,
        good_policy,
        adv_policy,
        benchmark,
        restore,
        load_dir,
        save_dir,
        lr,
        gamma,
        batch_size,
        num_units,
        display,
    ):
        self.scenario = scenario
        self.max_episode_len = max_episode_len
        self.num_episodes = num_episodes
        self.num_adversaries = num_adversaries
        self.good_policy = good_policy
        self.adv_policy = adv_policy
        self.benchmark = benchmark
        self.restore = restore
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_units = num_units
        self.display = display


if __name__ == "__main__":
    with U.single_threaded_session():
        arglist = MyObject(
            "",
            25,
            60000,
            3,
            "maddpg",
            "maddpg",
            False,
            False,
            "",
            "",
            1e-2,
            0.95,
            1024,
            64,
            False,
        )
        scenario = "tag_scenario2"
        goodAgentPolicy = "./policy-tag_scenario_base_2-60000/"
        advAgentPolicy = "./policy-tag_scenario2-60000/"

        env = make_env(scenario, arglist, False)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        U.initialize()

        if arglist.display or arglist.restore or arglist.benchmark or True:
            print("Loading previous state...")
            U.load_state(goodAgentPolicy)

        s = U.get_session()

        goodAgentVars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="agent_3"
        )
        advAgentsVars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="agent_0"
        ) + tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="agent_1"
        ) + tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="agent_2"
        )

        # for v in goodAgentVars:
        #     print(v.name)

        saver = tf.compat.v1.train.Saver(advAgentsVars)
        U.load_state(advAgentPolicy, saver=saver)

        newSaveDir = "./z_test/"

        U.save_state(newSaveDir)
