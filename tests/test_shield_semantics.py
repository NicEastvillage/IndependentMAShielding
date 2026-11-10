"""Unit tests for the synthesis logic on tiny, hand-checked transition systems.

These pin the semantics of :func:`imashielding.shield.compute_shield` without
using the road scenarios, so a regression can be diagnosed even if it only
shows up in the fixpoint or release handling.

Run from ``IndependentMAShielding`` with::

    python -m unittest discover -s tests
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from imashielding.shield import DGShield, compute_shield
from imashielding.systems import LabelledTransitionSystem, StateSafety

try:
    from shield_reference import assert_matches_reference
except ImportError:  # pragma: no cover - allows running as ``tests.test_...``
    from tests.shield_reference import assert_matches_reference


class ToySystem(LabelledTransitionSystem[str]):
    """A tiny labelled transition system.

    ``transitions[(state, action)]`` is the tuple of nondeterministic successors
    of ``action`` in ``state``.  Every state has ``action_count`` actions.
    """

    def __init__(self, init, transitions, action_count=1):
        self._init = init
        self._transitions = dict(transitions)
        self._action_count = action_count

    def get_init_state(self):
        return self._init

    def get_abstract_successor_states(self, state, action):
        return self._transitions[(state, action)]

    def get_action_count(self, state):
        return self._action_count


def safety_from(verdicts):
    """A safety predicate backed by an explicit state -> StateSafety mapping."""

    def safe_func(state):
        return verdicts[state]

    return safe_func


def compute(system, safety):
    # Avoid stderr/stdout noise from the synthesizer while tests run.
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_shield(system, safety)


class FixpointTest(unittest.TestCase):
    def test_all_actions_are_returned_when_all_successors_are_safe(self):
        system = ToySystem(
            "s0",
            {
                ("s0", 0): ("s1",),
                ("s0", 1): ("s1",),
                ("s0", 2): ("s1",),
                ("s1", 0): ("s1",),
                ("s1", 1): ("s1",),
                ("s1", 2): ("s1",),
            },
            action_count=3,
        )
        shield = compute(system, safety_from({"s0": StateSafety.SAFE, "s1": StateSafety.SAFE}))
        self.assertEqual(shield.get_safe_actions("s0"), (0, 1, 2))
        self.assertEqual(shield.get_safe_actions("s1"), (0, 1, 2))

    def test_action_leading_to_an_unsafe_state_is_removed(self):
        system = ToySystem(
            "s0",
            {
                ("s0", 0): ("good",),
                ("s0", 1): ("bad",),
                ("good", 0): ("good",),
                ("good", 1): ("good",),
            },
            action_count=2,
        )
        safety = safety_from(
            {"s0": StateSafety.SAFE, "good": StateSafety.SAFE, "bad": StateSafety.UNSAFE}
        )
        shield = compute(system, safety)
        self.assertEqual(shield.get_safe_actions("s0"), (0,))
        self.assertEqual(shield.get_safe_actions("good"), (0, 1))

    def test_a_state_with_no_safe_action_is_unsafe_and_poisons_predecessors(self):
        system = ToySystem(
            "s0",
            {
                ("s0", 0): ("s1",),
                ("s0", 1): ("s1",),
                ("s1", 0): ("dead",),
                ("s1", 1): ("dead",),
            },
            action_count=2,
        )
        safety = safety_from(
            {"s0": StateSafety.SAFE, "s1": StateSafety.SAFE, "dead": StateSafety.UNSAFE}
        )
        shield = compute(system, safety)
        self.assertEqual(shield.get_safe_actions("s0"), ())
        self.assertEqual(shield.get_safe_actions("s1"), ())
        self.assertEqual(shield.metadata["unsafe_states"], 3)  # s1, s0, dead
        self.assertEqual(shield.metadata["safe_states"], 0)

    def test_greatest_fixpoint_keeps_a_productive_cycle_alive(self):
        # s0 <-> s1 is a cycle, and both states have an escape into an unsafe
        # state.  Neither state is removed because action 0 can stay in the
        # cycle forever: this must be the *greatest* fixpoint, not the least.
        system = ToySystem(
            "s0",
            {
                ("s0", 0): ("s1",),
                ("s0", 1): ("bad",),
                ("s1", 0): ("s0",),
                ("s1", 1): ("bad",),
            },
            action_count=2,
        )
        safety = safety_from(
            {"s0": StateSafety.SAFE, "s1": StateSafety.SAFE, "bad": StateSafety.UNSAFE}
        )
        shield = compute(system, safety)
        self.assertEqual(shield.get_safe_actions("s0"), (0,))
        self.assertEqual(shield.get_safe_actions("s1"), (0,))

    def test_unsafe_successor_order_does_not_matter(self):
        for successors in (("bad", "good"), ("good", "bad")):
            with self.subTest(successors=successors):
                system = ToySystem(
                    "safe-action",
                    {
                        ("safe-action", 0): successors,
                        ("safe-action", 1): ("good",),
                        ("good", 0): ("good",),
                        ("good", 1): ("good",),
                    },
                    action_count=2,
                )
                safety = safety_from(
                    {
                        "safe-action": StateSafety.SAFE,
                        "good": StateSafety.SAFE,
                        "bad": StateSafety.UNSAFE,
                    }
                )
                shield = compute(system, safety)
                self.assertEqual(shield.get_safe_actions("safe-action"), (1,))

    def test_nondeterministic_successor_sets_are_all_required(self):
        system = ToySystem(
            "s0",
            {
                ("s0", 0): ("s1", "s2"),
                ("s0", 1): ("s1",),
                ("s1", 0): ("s1",),
                ("s1", 1): ("s1",),
            },
            action_count=2,
        )
        safety = safety_from(
            {"s0": StateSafety.SAFE, "s1": StateSafety.SAFE, "s2": StateSafety.UNSAFE}
        )
        shield = compute(system, safety)
        self.assertEqual(shield.get_safe_actions("s0"), (1,))


class ReleaseTest(unittest.TestCase):
    def test_release_state_is_terminal_and_not_expanded(self):
        system = ToySystem(
            "s0",
            {
                ("s0", 0): ("release",),
                ("release", 0): ("never-explored",),
            },
            action_count=1,
        )
        safety = safety_from({"s0": StateSafety.SAFE, "release": StateSafety.SAFE_RELEASE})
        shield = compute(system, safety)

        self.assertEqual(shield.get_safe_actions("s0"), (0,))
        self.assertTrue(shield.contains("release"))
        # A release state is terminal, so it has no safe actions to explore.
        self.assertTrue(shield.is_released("release"))
        # A release state must never be queried while every agent is shielded,
        # so asking for its actions is a loud failure rather than an empty mask.
        with self.assertRaises(AssertionError):
            shield.get_safe_actions("release")
        self.assertFalse(shield.contains("never-explored"))
        self.assertEqual(shield.metadata["release_states"], 1)
        # Metadata's safe count includes the release state (which is coloured
        # SAFE); release_states is an additional counter, not a disjoint set.
        self.assertEqual(shield.metadata["safe_states"], 2)

    def test_a_release_successor_does_not_mask_an_unsafe_successor(self):
        # Action 0 releases and action 1 fails; the union must still leave the
        # state safe *and* remove action 1.
        system = ToySystem(
            "s0",
            {
                ("s0", 0): ("release",),
                ("s0", 1): ("bad",),
            },
            action_count=2,
        )
        safety = safety_from(
            {
                "s0": StateSafety.SAFE,
                "release": StateSafety.SAFE_RELEASE,
                "bad": StateSafety.UNSAFE,
            }
        )
        shield = compute(system, safety)
        self.assertEqual(shield.get_safe_actions("s0"), (0,))
        self.assertEqual(shield.metadata["release_states"], 1)

    def test_release_at_a_successor_keeps_the_state_safe_even_without_a_cycle(self):
        system = ToySystem(
            "s0",
            {
                ("s0", 0): ("release",),
                ("s0", 1): ("release",),
            },
            action_count=2,
        )
        safety = safety_from({"s0": StateSafety.SAFE, "release": StateSafety.SAFE_RELEASE})
        shield = compute(system, safety)
        self.assertEqual(shield.get_safe_actions("s0"), (0, 1))


class ReferenceAgreementTest(unittest.TestCase):
    """Cross-check the hand-written cases against the reference oracle."""

    def test_hand_written_cases_agree_with_reference(self):
        system = ToySystem(
            "s0",
            {
                ("s0", 0): ("s1", "release"),
                ("s0", 1): ("bad",),
                ("s1", 0): ("s0",),
                ("s1", 1): ("bad",),
            },
            action_count=2,
        )
        safety = safety_from(
            {
                "s0": StateSafety.SAFE,
                "s1": StateSafety.SAFE,
                "release": StateSafety.SAFE_RELEASE,
                "bad": StateSafety.UNSAFE,
            }
        )
        shield = compute(system, safety)
        assert_matches_reference(self, system, safety, shield)


class ReleasePersistenceTest(unittest.TestCase):
    def test_release_state_still_raises_after_a_cache_round_trip(self):
        system = ToySystem(
            "s0",
            {
                ("s0", 0): ("release",),
                ("release", 0): ("never-explored",),
            },
            action_count=1,
        )
        safety = safety_from({"s0": StateSafety.SAFE, "release": StateSafety.SAFE_RELEASE})
        shield = compute(system, safety)

        with tempfile.TemporaryDirectory(prefix="imashielding-release-") as tmp:
            path = Path(tmp) / "shield.pkl.gz"
            shield.save(path)
            loaded = DGShield.load(path)

        self.assertEqual(loaded.get_safe_actions("s0"), (0,))
        self.assertTrue(loaded.contains("release"))
        with self.assertRaises(AssertionError):
            loaded.get_safe_actions("release")


if __name__ == "__main__":
    unittest.main()
