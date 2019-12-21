from dataclasses import dataclass
from typing import List, Optional
import abc
import random
import copy
import bisect
import itertools


Reward = str
Action = str


class Environment:
    def action_space(self):
        pass

    def copy_inplace(self, target: Optional['Environment']):
        pass


@dataclass
class Observation:
    env: Environment
    actions: List[Action]  # 他のプレーヤーのaction


RSPActions = ["R", "S", "P"]


class RSP(object):
    # Rock, Siccosrs, Paper
    rounds: int

    def __init__(self, rounds: int):
        self.rounds = rounds
        super().__init__()

    def action_space(self) -> List[Action]:
        return RSPActions

    def copy_inplace(self, target: Optional[Environment]):
        if not target:
            target = copy.deepcopy(self)
        target.rounds = self.rounds
        return target

    def step(self, actions) -> (List[Reward], bool):
        # 各agentのスコアを返す
        self.rounds -= 1
        assert(len(actions) == 2)

        return (self.calc_reward(actions), self.rounds == 0)

    def calc_reward(self, actions) -> List[Reward]:
        # actionを実行したときのrewardを計算する stepはしない
        reward1 = self.__get_reward_againts(actions[0], actions[1])
        return [reward1, -reward1]

    def __get_reward_againts(self, act1, act2) -> int:
        # R S P
        id1 = RSPActions.index(act1)
        id2 = RSPActions.index(act2)
        mod = (id2 - id1) % 3
        if mod == 0:
            return 0
        elif mod == 1:
            return 1
        elif mod == 2:
            return -1

        raise ValueError("action should be R, S or P.")


class Agent:
    def __init__(self, index: int, name: str) -> int:
        self.index = index
        self.name = name

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        pass


class ManualAgent(Agent):
    def __init__(self, index: int):
        super().__init__(index, "You")

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        while True:
            print(f"type your action from: {env.action_space()}")
            action = input().upper()
            if action in env.action_space():
                break
            print(f'{action} is NOT in action space.')
        return action


class RandomAgent(Agent):
    def __init__(self, index):
        super().__init__(index, "random_agent")

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        return random.choice(env.action_space())


class FirstAgent(Agent):
    def __init__(self, index):
        super().__init__(index, "first_action")

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        return env.action_space()[0]


class RegretMatchingAgent(Agent):
    old_env: Optional[Environment]

    def __init__(self, index):
        super().__init__(index, "regret_matching")
        self.regrets = None
        self.total_reward = 0
        self.old_env = None

    def act(self, reward, env: Environment, agent_actions: List[Action]) -> Action:
        self.total_reward += reward
        if not self.regrets:
            self.regrets = [0] * len(env.action_space())

        if self.old_env:
            for action_i, action in enumerate(self.old_env.action_space()):

                cf_actions = copy.copy(agent_actions)
                cf_actions[self.index] = action

                regret = max(0, self.old_env.calc_reward(cf_actions)[self.index] - reward)
                self.regrets[action_i] += regret
                action = self.random_pick(self.regrets, self.old_env.action_space() )

        else:
            action = random.choice(env.action_space())

        self.old_env = env.copy_inplace(self.old_env)

        return action

    def random_pick(self, weights, actions):
        ac = list(itertools.accumulate(weights))
        if ac[-1] > 0:
            f = lambda : bisect.bisect(ac, random.randint(0, ac[-1]-1))
            return actions[f()]
        else:
            return random.choice(actions)


def gameplay(env: Environment, agents: List[Agent]):
    reward = [0]*len(agents)
    total_reward = [0]*len(agents)
    actions = None
    while True:
        actions = [agent.act(reward, env, actions)
                   for agent, reward in zip(agents, reward)]
        reward, done = env.step(actions)

        for i in range(len(agents)):
            total_reward[i] += reward[i]

        print(', '.join([f'{agent.name}: {action} -> {tr}' for action,
                         agent, tr in zip(actions, agents, total_reward)]))

        if done:
            for agent, tr in zip(agents, total_reward):
                print(f'Agent:{agent.name} with reward={tr}')
            break


def init_agents(clss):
    return [a(i) for i, a in enumerate(clss)]

if __name__ == "__main__":
    env = RSP(rounds=100)
    #gameplay(env, init_agents([RandomAgent, FirstAgent]))
    gameplay(env, init_agents([RegretMatchingAgent, FirstAgent]))
    # gameplay(env, [ManualAgent(), FirstAgent()])
    # gameplay(env, [RandomAgent(0), FirstAgent(1)])
    #gameplay(env, [ManualAgent(), FirstAgent()])

def random_pick(weights, actions):
    ac = list(itertools.accumulate(weights))
    f = lambda : bisect.bisect(ac, random.randint(0, ac[-1]-1))
    return actions[f()]