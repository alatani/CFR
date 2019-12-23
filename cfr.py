import datetime
from dataclasses import dataclass
import pickle
from typing import List, Optional, Dict, Any
import abc
import cProfile
import sys
import random
import copy
import bisect
import itertools
from collections import defaultdict


Reward = int
Action = str


class Environment:
    def action_space(self, agent_index):
        pass

    def information_set(self, agent_index: int) -> str:
        pass

    def step(self, actions) -> (List[Reward], bool):
        pass


@dataclass
class GJState():
    n_cards: Dict[Action, int]
    stars: int

    def use_card(self, action: Action):
        assert(self.n_cards[action] > 0)
        self.n_cards[action] -= 1

    def restore_card(self, action: Action):
        self.n_cards[action] += 1

    def cards(self) -> int:
        return sum(n_cards.values())

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

    def is_terminal(self):
        return not self.alive()

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

    def information_set(self, agent_index: int):
        res = []
        assert(agent_index == 0 or agent_index == 1)

        counterpart_index = 1 - agent_index

        my_state = self.agent_states[agent_index]
        counterpart_state = self.agent_states[counterpart_index]
        return (
            # 自分
            my_state.n_cards['R'],
            my_state.n_cards['S'],
            my_state.n_cards['P'],
            my_state.stars,
            # 相手
            sum(counterpart_state.n_cards.values()),  # 枚数だけわかる
            counterpart_state.stars,
        )

    def action_space(self, agent_index: int) -> List[Action]:
        state = self.agent_states[agent_index]
        return state.get_usable_rsp()

    def backtrack(self, actions) -> (List[Reward], bool):
        for states, action in zip(self.agent_states, actions):
            states.restore_card(action)
        # 勝利判定
        hands = ["R", "S", "P"]
        id1 = hands.index(actions[0])
        id2 = hands.index(actions[1])
        mod = (id2 - id1) % 3
        r = 1
        if mod == 1:
            self.agent_states[0].lose()
            self.agent_states[1].win()
        elif mod == 2:
            self.agent_states[0].win()
            self.agent_states[1].lose()

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
            print(f"type your action from: {env.action_space(self.index)}")
            action = input().upper()
            if action in env.action_space(self.index):
                break
            print(f'{action} is NOT in action space.')
        return action

    def update_policy(self, reward, agent_actions: List[Action]):
        pass


class RandomAgent(Agent):
    def __init__(self, index):
        super().__init__(index, "random_agent")

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        acs = env.action_space(self.index)
        if acs:
            return random.choice(acs)
        else:
            return None

    def update_policy(self, reward, agent_actions: List[Action]):
        pass

    def repr_state(self):
        return "random"


class FirstAgent(Agent):
    def __init__(self, index):
        super().__init__(index, "first_action")

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        return env.action_space(self.index)[0]


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
            def f(): return bisect.bisect(ac, random.random()*ac[-1])
            return actions[f()]
        else:
            return random.choice(actions)


class CFR(Agent):
    old_env: Optional[Environment]

    cum_regrets: Dict[Any, float]  # (info_set, Action) -> regret
    cum_strategy: Dict[Any, int]  # (info_set, Action) -> regret
    profile: Dict[Any, float]

    def __init__(self, index):
        super().__init__(index, f"CFR({index})")

        self.cum_regrets = defaultdict(float)
        self.cum_strategy = defaultdict(int)
        self.profile = {}

        self.counterpart_index = 1 - index

    def train(self, env: Environment, T: int):
        for t in range(T):
            for index in [0, 1]:
                self.__cfr(env, index, t, 1.0, 1.0, 0)
        return self

    def __get_profile(self, I, action, action_space):
        if (I, action) not in self.profile:
            prof = 1.0 / len(action_space)
            self.profile[(I, action)] = prof
            return prof
        else:
            return self.profile[(I, action)]

    def __cfr(self, env: Environment, index, t, pi_me, pi_c, depth):
        # 自分のターン
        if depth == 1:
            print("First iteration:", env.agent_states, datetime.datetime.now())
        if depth == 2:
            print("    Second iteration:",
                  env.agent_states, datetime.datetime.now())
        if depth == 3:
            print("        Third iteration:",
                  env.agent_states, datetime.datetime.now())
        counterpart_index = 1 - index
        action_space = env.action_space(index)
        vs = [0.0] * len(action_space)
        v = 0.0

        I = env.information_set(index)
        search_node = random.randint(0, len(action_space)-1)
        for i, a in enumerate(action_space):
            prof = self.__get_profile(I, a, action_space)
            vs[i] = self.__cfr_counterpart(
                a, env, counterpart_index, t, prof * pi_me, pi_c, depth)

            v += prof*vs[i]

        for i, a in enumerate(action_space):
            prof = self.__get_profile(I, a, action_space)
            self.cum_regrets[(I, a)] += pi_c * (vs[i] - v)
            self.cum_strategy[(I, a)] += pi_me * prof
        self.__update_profiles(I, action_space)
        return v

    def __cfr_counterpart(self, my_action, env: Environment, counterpart_index, t, pi_me, pi_c, depth):
        # 相手のターン
        index = 1 - counterpart_index
        action_space = env.action_space(counterpart_index)

        actions = [None, None]
        actions[index] = my_action
        vs = [0.0] * len(action_space)
        v = 0.0
        I = env.information_set(counterpart_index)
        for i, a in enumerate(action_space):
            actions[counterpart_index] = a

            # new_env = env
            new_env = copy.deepcopy(env)
            rewards, done = new_env.step(actions)

            Inext = new_env.information_set(counterpart_index)

            prof = self.__get_profile(I, a, action_space)
            if done:
                # terminal node
                vs[i] = rewards[index]
            else:
                vs[i] = self.__cfr(new_env, index, t, pi_me,
                                   prof*pi_c, depth+1)
            v += prof * vs[i]
            # new_env.backtrack(actions)
        return v

    def __update_profiles(self, I, action_space):
        denominator = 0.0
        for a in action_space:
            denominator += self.cum_regrets[(I, a)]
        if denominator > 0.0001:
            for a in action_space:
                self.profile[(I, a)] = self.cum_regrets[(I, a)] / denominator
        else:
            d = 1.0 / len(action_space)
            for a in action_space:
                self.profile[(I, a)] = d

    def __sample_action(self, agent_index, h: Environment, debug=False) -> Action:
        I = h.information_set(agent_index)
        action_space = h.action_space(agent_index)
        profiles = []
        for a in action_space:
            if (I, a) in self.profile:
                prof = self.profile[(I, a)]
            else:
                prof = 1.0 / len(action_space)
                # print("anaume")
                # init profile if not set yet
                self.profile[(I, a)] = prof
            profiles.append(prof)
        picked = self.random_pick(profiles, action_space)

        if debug:
            print(f'picked {picked} from {action_space}.  prof={profiles}. key={(I, picked)}')
        return picked

    def repr_state(self):
        pass

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        return self.__sample_action(self.index, env, True)
        # return action

    def update_policy(self, reward, agent_actions: List[Action]):
        pass

    def random_pick(self, weights, actions):
        ac = list(itertools.accumulate(weights))
        if ac[-1] > 0:
            def f(): return bisect.bisect(ac, random.random()*ac[-1])
            return actions[f()]
        else:
            return random.choice(actions)


def gameplay(env_gen, agents: List[Agent], games: int):
    total_reward = [0]*len(agents)

    for t in range(games):
        env = env_gen()
        actions = None
        rewards = [0]*len(agents)
        while True:
            actions = [agent.act(reward, env, actions) for agent, reward in zip(agents, rewards)]

            rewards, done = env.step(actions)

            for reward, agent in zip(rewards, agents):
                agent.update_policy(reward, actions)

            for i in range(len(agents)):
                total_reward[i] += rewards[i]
            if done:
                break

        match_result = ', '.join(
            [f'{agent.name}: {tr}' for agent, tr in zip(agents, total_reward)]
        )
        print(match_result)

    # match_result = ', '.join(
    #     [f'{agent.name}: {action} -> {tr}' for action,
    #         agent, tr in zip(actions, agents, total_reward)]
    # )
    # # print(f'{match_result}  {str(agents[1].repr_state())}')
    # print(f'{match_result} ')


def init_agents(clss):
    return [a(i) for i, a in enumerate(clss)]


if __name__ == "__main__2":
    with open("gj2222.pickle", "rb") as f:
        cfr = pickle.load(f)
        print(cfr.profile)

if __name__ == "__main__":

    print("now=", datetime.datetime.now())
    # env = RSP(["R", "S", "P"], rounds=10000)

    # env = GJ(GJState({'R': 2, 'S': 2, 'P': 2}, stars=2))
    # cfr = CFR(1).train(env, T=2)
    # with open("gj2222.pickle", "wb") as f:
    #     pickle.dump(cfr, f)

    # env = GJ(GJState({'R': 4, 'S': 4, 'P': 4}, stars=4))
    # with open("gj4444.pickle", "wb") as f:
    #     pickle.dump(env, f)
    # env = GJ(GJState({'R': 3, 'S': 3, 'P': 3}, stars=3))
    # with open("gj3333.pickle", "wb") as f:
    #     pickle.dump(env, f)
    # env = GJ(GJState({'R': 1, 'S': 1, 'P': 1}, stars=1))
    # with open("gj1111.pickle", "wb") as f:
    #     pickle.dump(env, f)

    # env = GJ(GJState({'R': 3, 'S': 3, 'P': 3}, stars=3))
    # cfr = CFR(1).train(env, T=1)
    # with open("gj3333.pickle", "wb") as f:
    #     pickle.dump(cfr, f)

    # env = GJ(GJState({'R': 2, 'S': 2, 'P': 2}, stars=2))
    # cfr = CFR(1).train(env, T=1)
    # with open("gj2222.pickle", "wb") as f:
    #     pickle.dump(cfr, f)

    # env = GJ(GJState({'R': 4, 'S': 4, 'P': 4}, stars=3))
    # cfr = CFR(1).train(env, T=1)
    # with open("gj4443.pickle", "wb") as f:
    #     pickle.dump(cfr, f)
    # sys.exit()

    # with open("gj3333.pickle", "rb") as f:
    #     cfr = pickle.load(f)
    with open("gj2222.pickle", "rb") as f:
        cfr = pickle.load(f)
    print(cfr)

    print("============== FINISHED TRAINING ==============")
    cfr.index = 1
    # agents = [ManualAgent(0), cfr]
    agents = [RandomAgent(0), cfr]
    #agents = [RandomAgent(0), RandomAgent(1)]
    # agents = [RandomAgent(0), RandomAgent(1)]
    # env = GJ(GJState.single())

    # def env_gen(): return GJ(GJState({'R': 3, 'S': 3, 'P': 3}, stars=3))
    def env_gen(): return GJ(GJState({'R': 2, 'S': 2, 'P': 2}, stars=2))
    gameplay(env_gen, agents, games=10000)
