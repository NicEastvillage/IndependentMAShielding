"""Randomised cross-check of :func:`compute_shield` against the reference oracle.

The hand-written cases pin individual behaviours; this test generates many
small transition systems with random nondeterminism and random SAFE / UNSAFE /
SAFE_RELEASE labels.  It is deterministic (fixed seed) and cheap, and catches
fixpoint or release-handling changes that the hand-written cases miss.

Run from ``IndependentMAShielding`` with::

    python -m unittest discover -s tests
"""

import contextlib
import io
import random
import unittest

from imashielding.shield import compute_shield
from imashielding.systems import LabelledTransitionSystem, StateSafety

try:
    from shield_reference import assert_matches_reference
except ImportError:  # pragma: no cover - allows running as ``tests.test_...``
    from tests.shield_reference import assert_matches_reference


class RandomSystem(LabelledTransitionSystem[str]):
    def __init__(self, init, transitions, action_count):
        self._init = init
        self._transitions = transitions
        self._action_count = action_count

    def get_init_state(self):
        return self._init

    def get_abstract_successor_states(self, state, action):
        return self._transitions[(state, action)]

    def get_action_count(self, state):
        return self._action_count


def _random_system(rng, state_count=6, action_count=3):
    states = [f"s{i}" for i in range(state_count)]
    transitions = {}
    for state in states:
        for action in range(action_count):
            successor_count = rng.randint(1, 3)
            transitions[(state, action)] = tuple(
                rng.choice(states) for _ in range(successor_count)
            )

    # The initial state must be SAFE; the synthesizer (like the paper) assumes
    # a safe start.  Other states are mostly safe, with enough unsafe and
    # release states that both paths are exercised.
    verdicts = {"s0": StateSafety.SAFE}
    for state in states[1:]:
        roll = rng.random()
        if roll < 0.25:
            verdicts[state] = StateSafety.UNSAFE
        elif roll < 0.40:
            verdicts[state] = StateSafety.SAFE_RELEASE
        else:
            verdicts[state] = StateSafety.SAFE

    def safety(state):
        return verdicts[state]

    return RandomSystem("s0", transitions, action_count), safety


class RandomReferenceAgreementTest(unittest.TestCase):
    def test_random_systems_match_reference(self):
        rng = random.Random(20261105)
        for index in range(150):
            with self.subTest(case=index):
                system, safety = _random_system(rng)
                with contextlib.redirect_stdout(io.StringIO()):
                    shield = compute_shield(system, safety)
                assert_matches_reference(self, system, safety, shield)


if __name__ == "__main__":
    unittest.main()
