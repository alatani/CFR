
from dataclasses import dataclass
from typing import List, Optional, Dict
import abc
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
    for _ in range(10):
        print(random_pick([0.2, 0.7, 0.5], ["a", "b", "c"]))
