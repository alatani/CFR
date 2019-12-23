from dataclasses import dataclass
from typing import List, Optional, Dict
import abc
import random
import copy
import bisect
import itertools
from collections import defaultdict


Reward = str
Action = str


class Environment:
    def action_space(self):
        pass


@dataclass
class GJState():
    n_cards: Dict[Action, int]
    stars: int

    def use_card(self, action: Action):
        assert(self.n_cards[action] > 0)
        self.n_cards[action] -= 1

    def win(self):
        self.stars += 1

    def lose(self):
        self.stars -= 1

    def alive(self):
        if self.stars <= 0:
            return False
        for v in self.n_cards.values():
            if v > 0:
                return True
        return False

    def get_usable_rsp(self) -> List[Action]:
        res = []
        for action in ["R", "S", "P"]:
            if self.n_cards[action] > 0:
                res.append(action)
        return res

    @classmethod
    def single(cls) -> 'GJState':
        return GJState({'R': 1, 'S': 1, 'P': 1}, 1)

    @classmethod
    def kaiji(cls, stars=4) -> 'GJState':
        return GJState({'R': 4, 'S': 4, 'P': 4}, stars)


class GJ(Environment):

    agent_states: List[GJState]

    def __init__(self, init_state: GJState):
        super().__init__()
        self.agent_states = [copy.deepcopy(init_state) for _ in range(2)]

    def action_space(self, agent_index: int) -> List[Action]:
        state = self.agent_states[agent_index]
        return state.get_usable_rsp()

    def step(self, actions) -> (List[Reward], bool):
        # 各agentのスコアを返す
        assert(len(actions) == 2)

        # 使用したカードを減らす
        for states, action in zip(self.agent_states, actions):
            states.use_card(action)

        # 勝利判定
        hands = ["R", "S", "P"]
        id1 = hands.index(actions[0])
        id2 = hands.index(actions[1])
        mod = (id2 - id1) % 3
        r = 1
        if mod == 1:
            self.agent_states[0].win()
            self.agent_states[1].lose()
            [1, -1]
        elif mod == 2:
            self.agent_states[0].lose()
            self.agent_states[1].win()

        rewards = self.calc_reward(actions)
        # if rewards[0] == 0 and rewards[1] == 0:
        if self.agent_states[0].alive() and self.agent_states[1].alive():
            return rewards, False
        else:
            return rewards, True

    def calc_reward(self, actions) -> List[Reward]:
        if self.agent_states[0].stars > 0 and self.agent_states[1].stars == 0:
            return [1, -1]
        elif self.agent_states[0].stars == 0 and self.agent_states[1].stars > 0:
            return [-1, 1]
        else:
            return [0, 0]


@dataclass
class Observation:
    env: Environment
    actions: List[Action]  # 他のプレーヤーのaction


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
        super().__init__(index, f"regret_matching({index})")
        self.regrets = defaultdict(int)
        self.total_reward = 0
        self.old_env = None

    def repr_state(self):
        sum_of_regrets = max(sum(self.regrets.values() or [0]), 0.1)

        res = {}
        for k, v in self.regrets.items():
            res[k] = v / sum_of_regrets
        return res

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        self.total_reward += reward

        action_space = env.action_space(self.index)
        # if not action_space:
        #     return None
        if self.old_env:
            weights = [self.regrets[action]
                       for action in action_space]
            action = self.random_pick(
                weights, env.action_space(self.index)
            )

        else:
            action = random.choice(action_space)

        self.old_env = copy.deepcopy(env)

        return action

    def update_policy(self, reward, agent_actions: List[Action]):
        if self.old_env:
            for action in self.old_env.action_space(self.index):
                cf_actions = copy.copy(agent_actions)
                cf_actions[self.index] = action

                regret = max(0, self.old_env.calc_reward(
                    cf_actions)[self.index] - reward)
                self.regrets[action] += regret

    def random_pick(self, weights, actions):
        ac = list(itertools.accumulate(weights))
        if ac[-1] > 0:
            def f(): return bisect.bisect(ac, random.randint(0, ac[-1]-1))
            return actions[f()]
        else:
            return random.choice(actions)


def gameplay(env: Environment, agents: List[Agent]):
    rewards = [0]*len(agents)
    total_reward = [0]*len(agents)
    actions = None
    while True:
        actions = [agent.act(reward, env, actions)
                   for agent, reward in zip(agents, rewards)]

        rewards, done = env.step(actions)

        for reward, agent in zip(rewards, agents):
            agent.update_policy(reward, actions)

        for i in range(len(agents)):
            total_reward[i] += rewards[i]


        if done:
            break

    match_result = ', '.join(
        [f'{agent.name}: {action} -> {tr}' for action, agent, tr in zip(actions, agents, total_reward)]
    )
    print(f'{match_result}  {str(agents[0].repr_state())}')


def init_agents(clss):
    return [a(i) for i, a in enumerate(clss)]


if __name__ == "__main__":
    # env = RSP(["R", "S", "P"], rounds=10000)


    agents = init_agents([RegretMatchingAgent, RegretMatchingAgent])
    for i in range(300):
        # env = GJ(GJState.single())
        env = GJ(GJState.kaiji(stars=2))
        gameplay(env, agents)
