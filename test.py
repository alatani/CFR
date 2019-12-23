
from dataclasses import dataclass
from typing import List, Optional, Dict
import abc
import pickle
import random
import copy
import bisect
import itertools
from collections import defaultdict


def random_pick(weights, actions):
    ac = list(itertools.accumulate(weights))
    if ac[-1] > 0:
        def f(): return bisect.bisect(ac, random.random()*ac[-1])
        return actions[f()]
    else:
        return random.choice(actions)


if __name__ == "__main__":
    with open("gj2222.pickle", "rb") as f:
        cfr = pickle.load(f)
    print(cfr)
