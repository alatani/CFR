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
from collections import defaultdict, Counter


Reward = int
Action = str


class Environment:
    def action_space(self, agent_index):
        pass

    def information_set(self, agent_index: int) -> str:
        pass

    def step(self, index, action) -> bool:
        pass

    def is_terminal(self) -> bool:
        pass

    def get_reward(self, index) -> int:
        pass

    def turn_of(self) -> int:
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
        return not self.is_terminal()

    def is_terminal(self):
        if self.stars <= 0:
            return True
        for v in self.n_cards.values():
            if v > 0:
                return False
        return True

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

        self.actions = [None, None]
        self.__terminal = False
        self.turn = 0

    def turn_of(self):
        return self.turn

    def information_set(self, agent_index: int):
        assert(agent_index == 0 or agent_index == 1)

        counterpart_index = 1 - agent_index

        my_state = self.agent_states[agent_index]
        counterpart_state = self.agent_states[counterpart_index]
        return (
            # 自分
            (my_state.n_cards['R'],
             my_state.n_cards['S'],
             my_state.n_cards['P'],
             my_state.stars),
            # 相手
            (counterpart_state.n_cards['R'],  # 今まで何枚使ったかわかるはず
             counterpart_state.n_cards['S'],
             counterpart_state.n_cards['P'],
             counterpart_state.stars),
        )

    def action_space(self, agent_index: int) -> List[Action]:
        state = self.agent_states[agent_index]
        return state.get_usable_rsp()

    def step(self, index, action) -> bool:
        # 各agentのスコアを返す
        assert(index in [0, 1])

        # 使用したカードを減らす
        # for states, action in zip(self.agent_states, actions):
        #     states.use_card(action)
        state = self.agent_states[index]
        state.use_card(action)
        self.actions[index] = action

        if self.actions[0] and self.actions[1]:
            # 勝利判定
            hands = ["R", "S", "P"]
            id1 = hands.index(self.actions[0])
            id2 = hands.index(self.actions[1])
            mod = (id2 - id1) % 3
            r = 1
            if mod == 1:
                self.agent_states[0].win()
                self.agent_states[1].lose()
            elif mod == 2:
                self.agent_states[0].lose()
                self.agent_states[1].win()

        # isterminal = self.is_terminal()
        if self.actions[0] and self.actions[1]:
            self.__terminal = (self.agent_states[0].is_terminal(
            ) or self.agent_states[1].is_terminal())
        else:
            self.__terminal = False

        if index == 0:
            assert(self.turn == 0)
            self.turn = 1
        elif index == 1:
            assert(self.turn == 1)
            self.turn = 0
            self.actions = [None, None]
        return self.__terminal

    def is_terminal(self):
        return self.__terminal

    def get_reward(self, index) -> int:
        if self.is_terminal():
            counterpart = 1 - index
            my_stars = self.agent_states[index].stars
            your_stars = self.agent_states[counterpart].stars
            if my_stars > your_stars:
                return 1
            if my_stars < your_stars:
                return -1
            return 0
        return None


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


class RandomAgent(Agent):
    def __init__(self, index):
        super().__init__(index, "random_agent")

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        acs = env.action_space(self.index)
        if acs:
            return random.choice(acs)
        else:
            return None

    def repr_state(self):
        return "random"


class FirstAgent(Agent):
    def __init__(self, index):
        super().__init__(index, "first_action")

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        return env.action_space(self.index)[0]


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

    def __cfr(self, env: Environment, index, t, pi0, pi1, depth):
        if depth <= 2:
            print(" " * depth + f"{depth}-th iteration:", env.agent_states,
                f"actions:{env.actions} turn={env.turn_of()} terminal:{env.is_terminal()}  reward={env.get_reward(index)}(index={index})", datetime.datetime.now())

        if env.is_terminal():
            return env.get_reward(index)

        counterpart_index = 1 - index
        action_space = env.action_space(env.turn_of())
        vs = [0.0] * len(action_space)
        v = 0.0
        I = env.information_set(index)

        for i, a in enumerate(action_space):
            prof = self.__get_profile(I, a, action_space)

            new_env = copy.deepcopy(env)

            if env.turn_of() == 0:
                _ = new_env.step(0, a)
                vs[i] = self.__cfr(new_env, index, t,
                                   prof * pi0, pi1, depth+1)
            elif env.turn_of() == 1:
                # print(new_env.agent_states, a)
                _ = new_env.step(1, a)
                vs[i] = self.__cfr(new_env, index, t, pi0,
                                   prof * pi1, depth+1)
            else:
                raise ValueError()

            v += prof*vs[i]
        # if I == ((0, 1, 1, 1), (0, 1, 1, 1)) and index == 0:
        #     print("depth", depth)
        #     print("index", index)
        #     print("env", env.agent_states)
        #     print("action_space", action_space)
        #     print("vs", vs)
        #     print("v", v)
        #     print("turn", env.turn_of())

        my_pi = [pi0, pi1][index]
        your_pi = [pi0, pi1][counterpart_index]
        if env.turn_of() == index:
            for i, a in enumerate(action_space):
                prof = self.__get_profile(I, a, action_space)
                self.cum_regrets[(I, a)] += your_pi * (vs[i] - v)
                self.cum_strategy[(I, a)] += my_pi * prof
            self.__update_profiles(I, action_space)
        return v

    def __update_profiles(self, I, action_space):
        denominator = 0.0
        for a in action_space:
            denominator += max(self.cum_regrets[(I, a)], 0.0)
        if denominator > 0.0001:

            for a in action_space:
                prof = max(self.cum_regrets[(I, a)], 0.0) / denominator
                self.profile[(I, a)] = prof
        else:
            d = 1.0 / len(action_space)
            for a in action_space:
                self.profile[(I, a)] = d

    def __sample_action(self, agent_index, env: Environment, debug=False) -> Action:
        I = env.information_set(agent_index)
        action_space = env.action_space(agent_index)
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

        print(
            f'picked {picked} from {action_space}.  prof={profiles}. key={(I, picked)}')
        return picked

    def repr_state(self):
        pass

    def act(self, reward, env: Environment, actions: List[Action]) -> Action:
        return self.__sample_action(self.index, env)
        # return action

    def random_pick(self, weights, actions):
        ac = list(itertools.accumulate(weights))
        if ac[-1] > 0:
            def f(): return bisect.bisect(ac, random.random()*ac[-1])
            return actions[f()]
        else:
            return random.choice(actions)


def gameplay(env_gen, agents: List[Agent], games: int):
    total_reward = [0]*len(agents)
    result = Counter()

    for t in range(games):
        env = env_gen()
        actions = None
        rewards = [0]*len(agents)
        while True:
            actions = [agent.act(reward, env, actions)
                       for agent, reward in zip(agents, rewards)]

            for i, action in enumerate(actions):
                done = env.step(i, action)
                if done:
                    break

            for i in range(len(agents)):
                total_reward[i] += env.get_reward(i) or 0
                rewards[i] += env.get_reward(i) or 0
            if done:
                break

        result.update([rewards[1]])
        match_result = ', '.join(
            [f'{agent.name}: {tr}' for agent, tr in zip(agents, total_reward)]
        )
        print(match_result)
        print(result)

    # match_result = ', '.join(
    #     [f'{agent.name}: {action} -> {tr}' for action,
    #         agent, tr in zip(actions, agents, total_reward)]
    # )
    # # print(f'{match_result}  {str(agents[1].repr_state())}')
    # print(f'{match_result} ')


def init_agents(clss):
    return [a(i) for i, a in enumerate(clss)]


if __name__ == "__main__2":
    # Ir = (((0, 1, 1, 1), (0, 1, 1, 1)), 'R')
    Is = (((0, 1, 1, 1), (0, 1, 1, 1)), 'S')
    Ip = (((0, 1, 1, 1), (0, 1, 1, 1)), 'P')
    Is = [Is, Ip]
    with open("gj1111.pickle", "rb") as f:
        cfr = pickle.load(f)
        for I in Is:
            print(cfr.profile[I])
        print("---")
        for I in Is:
            print(cfr.cum_regrets[I])
        print("---")
        for I in Is:
            print(cfr.cum_strategy[I])
    sys.exit()

if __name__ == "__main__":

    print("now=", datetime.datetime.now())
    # env = RSP(["R", "S", "P"], rounds=10000)

    # env = GJ(GJState({'R': 1, 'S': 1, 'P': 1}, stars=1))
    # cfr = CFR(1).train(env, T=3)
    # with open("gj1111.pickle", "wb") as f:
    #     pickle.dump(cfr, f)
    #sys.exit()

    # env = GJ(GJState({'R': 2, 'S': 2, 'P': 2}, stars=2))
    # cfr = CFR(1).train(env, T=10)
    # with open("gj2222.pickle", "wb") as f:
    #     pickle.dump(cfr, f)

    # env = GJ(GJState({'R': 2, 'S': 2, 'P': 2}, stars=3))
    # cfr = CFR(1).train(env, T=10)
    # with open("gj2223.pickle", "wb") as f:
    #     pickle.dump(cfr, f)
    env = GJ(GJState({'R': 3, 'S': 3, 'P': 3}, stars=3))
    cfr = CFR(1).train(env, T=10)
    with open("gj2223.pickle", "wb") as f:
        pickle.dump(cfr, f)

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
    # cfr = CFR(1).train(env, T=10)
    # with open("gj2222.pickle", "wb") as f:
    #     pickle.dump(cfr, f)

    # env = GJ(GJState({'R': 4, 'S': 4, 'P': 4}, stars=3))
    # cfr = CFR(1).train(env, T=1)
    # with open("gj4443.pickle", "wb") as f:
    #     pickle.dump(cfr, f)
    # sys.exit()

    # with open("gj3333.pickle", "rb") as f:


    #     cfr = pickle.load(f)
    # with open("gj1111.pickle", "rb") as f:
    #     cfr = pickle.load(f)
    # cfr.index = 1
    # agents = [RandomAgent(0), cfr]
    # def env_gen(): return GJ(GJState({'R': 1, 'S': 1, 'P': 1}, stars=1))
    # gameplay(env_gen, agents, games=1000)

    # with open("gj2222.pickle", "rb") as f:
    #     cfr = pickle.load(f)
    # cfr.index = 1
    # agents = [RandomAgent(0), cfr]
    # def env_gen(): return GJ(GJState({'R': 2, 'S': 2, 'P': 2}, stars=2))
    # gameplay(env_gen, agents, games=10000)

    # def env_gen(): return GJ(GJState({'R': 3, 'S': 3, 'P': 3}, stars=3))
    #def env_gen(): return GJ(GJState({'R': 1, 'S': 1, 'P': 1}, stars=1))

    # with open("gj2223.pickle", "rb") as f:
    #     cfr = pickle.load(f)
    # cfr.index = 1
    # agents = [RandomAgent(0), cfr]
    # def env_gen(): return GJ(GJState({'R': 2, 'S': 2, 'P': 2}, stars=3))
    # gameplay(env_gen, agents, games=10000)
    with open("gj3333.pickle", "rb") as f:
        cfr = pickle.load(f)
    cfr.index = 1
    agents = [RandomAgent(0), cfr]
    def env_gen(): return GJ(GJState({'R': 2, 'S': 2, 'P': 2}, stars=3))
    gameplay(env_gen, agents, games=10000)