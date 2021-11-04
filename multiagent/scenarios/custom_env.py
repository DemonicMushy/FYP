import numpy as np

from multiagent.core import Agent, Landmark, World
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        num_good = 1
        num_adversaries = 3
        num_obstacles = 2
        good_sight_range = 5
        adversaries_sight_range = -1
        world = World()
        # set any world properties first
        world.dim_c = 0  # set the communication dimension
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.sight_range = (
                adversaries_sight_range if agent.adversary else good_sight_range
            )
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def withinSightRange(self, agent, other):
        if agent.sight_range == -1:
            # all seeing entity
            return True
        relative_pos = other.state.p_pos - agent.state.p_pos
        relative_dist_squared = abs(pow(relative_pos[0], 2) - pow(relative_pos[1], 2))

        if relative_dist_squared <= pow(agent.sight_range, 2):
            return True
        else:
            return False

    def adversaryNearby(self, good_agent, world):
        for adv in self.adversaries(world):
            if self.withinSightRange(good_agent, adv):
                return True

    def observation(self, agent, world):

        observation = (
            self.adversary_observation(agent, world)
            if agent.adversary
            else self.agent_observation(agent, world)
        )
        return observation

    def agent_observation(self, agent, world):
        # here is where each agent's observation will be
        # we can implement the agent's ability to 'listen' to certain agents here
        # blocked by landmarks?
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                if self.withinSightRange(agent, entity):
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        friendly_pos = []
        friendly_vel = []
        enemy_pos = []
        enemy_vel = []
        for other in world.agents:
            if other is agent:  # check for itself
                continue
            if not other.adversary:
                if self.withinSightRange(agent, other):
                    # check if other agent is within agent's sight range
                    friendly_pos.append(other.state.p_pos - agent.state.p_pos)
                    friendly_vel.append(other.state.p_vel)
            elif other.adversary:
                if self.withinSightRange(agent, other):
                    # check if other agent is within agent's sight range
                    enemy_pos.append(other.state.p_pos - agent.state.p_pos)
                    enemy_vel.append(other.state.p_vel)
        for other in self.good_agents(world):
            if other is agent:
                continue
            comm_temp = []
            comm_temp.append(other.state.p_pos - agent.state.p_pos)
            for adv in self.adversaries(world):
                if self.withinSightRange(other, adv):
                    comm_temp.append(adv.state.p_pos - other.state.p_pos)
            if len(comm_temp) > 0:
                comm.append(comm_temp)

        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + friendly_pos
            + friendly_vel  # relative positions of agents
            + enemy_pos  # velocity of friendly agents
            + enemy_vel
            # + comm
        )

    def adversary_observation(self, agent, world):
        # here is where each agent's observation will be
        # we can implement the agent's ability to 'listen' to certain agents here
        # blocked by landmarks?
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                if self.withinSightRange(agent, entity):
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        friendly_pos = []
        friendly_vel = []
        enemy_pos = []
        enemy_vel = []
        for other in world.agents:
            if other is agent:  # check for itself
                continue
            # comm.append(other.state.c)
            if other.adversary:
                if self.withinSightRange(agent, other):
                    # check if other agent is within agent's sight range
                    friendly_pos.append(other.state.p_pos - agent.state.p_pos)
                    friendly_vel.append(other.state.p_vel)
            elif not other.adversary:
                if self.withinSightRange(agent, other):
                    # check if other agent is within agent's sight range
                    enemy_pos.append(other.state.p_pos - agent.state.p_pos)
                    enemy_vel.append(other.state.p_vel)

        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + friendly_pos  # relative positions of agents
            + friendly_vel  # velocity of friendly agents
            + enemy_pos
            + enemy_vel
            # + comm
        )
