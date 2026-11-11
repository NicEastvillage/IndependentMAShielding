"""Regression tests for the road scenarios and the abstraction layer.

These are the cases the project actually cares about:
``AbstractRoundaboutSystem`` (the 2-car abstraction) and
``AbstractRoadNetworkSystem`` on the crossing-roads and double-roundabout
scenarios.  Each synthesized shield is compared against the independent
reference oracle in ``shield_reference.py``.

Run from ``IndependentMAShielding`` with::

    python -m unittest discover -s tests
"""

import contextlib
import io
import tempfile
import unittest

from imashielding.roads import (
    CarDef,
    CarState,
    ChancedOutcomeAccel,
    RoadNetwork,
    RoadNetworkState,
    RoadSegment,
)
from imashielding.roads_reduction import (
    AbstractRoadNetworkSystem,
    AbstractRoundaboutSystem,
    road_safety,
    roundabout_safety,
)
from imashielding.shield import compute_shield
from imashielding.systems import StateSafety

try:
    from shield_reference import assert_matches_reference
except ImportError:  # pragma: no cover - allows running as ``tests.test_...``
    from tests.shield_reference import assert_matches_reference


def compute(system, safety, cache_dir=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_shield(system, safety, cache_dir=cache_dir)


def car(id, road=0, pos=0, vel=0, just_merged=False):
    return CarState(id, road, pos, vel, just_merged, False)


class RoundaboutSafetyTest(unittest.TestCase):
    def test_single_car_is_safe(self):
        safety = roundabout_safety(min_dist=2)
        self.assertIs(
            safety(RoadNetworkState((car(0, pos=0, vel=1),))), StateSafety.SAFE
        )

    def test_gap_below_min_dist_is_unsafe(self):
        safety = roundabout_safety(min_dist=3)
        state = RoadNetworkState((car(0, pos=0, vel=1), car(1, pos=2, vel=1)))
        self.assertIs(safety(state), StateSafety.UNSAFE)

    def test_gap_at_or_above_min_dist_is_safe(self):
        safety = roundabout_safety(min_dist=3)
        safe = RoadNetworkState((car(0, pos=0, vel=1), car(1, pos=3, vel=1)))
        self.assertIs(safety(safe), StateSafety.SAFE)

    def test_roundabout_safety_never_releases(self):
        safety = roundabout_safety(min_dist=3)
        for gap in range(0, 8):
            state = RoadNetworkState((car(0, pos=0), car(1, pos=gap)))
            self.assertNotEqual(safety(state), StateSafety.SAFE_RELEASE)


class RoundaboutNormalizationTest(unittest.TestCase):
    def test_normalize_projects_to_ego_and_nearest_car_ahead(self):
        network = RoadNetwork.create_roundabout_scenario(
            car_count=2, road_length=12, max_vel=2
        )
        system = AbstractRoundaboutSystem(network)
        state = RoadNetworkState(
            (
                car(0, road=0, pos=0, vel=1),
                car(1, road=0, pos=4, vel=2),
            )
        )
        ego_view = system.normalize(state, 0)
        self.assertEqual(len(ego_view.cars), 2)
        self.assertEqual(ego_view.cars[0].pos, 0)
        self.assertEqual(ego_view.cars[1].pos, 4)
        self.assertEqual(ego_view.cars[0].vel, 1)
        self.assertEqual(ego_view.cars[1].vel, 2)

        # From car 1's point of view car 0 is 12 - 4 = 8 cells ahead (a lap
        # around the ring), and car 1's own position is canonicalised to 0.
        other_view = system.normalize(state, 1)
        self.assertEqual(other_view.cars[0].pos, 0)
        self.assertEqual(other_view.cars[1].pos, 8)

    def test_roundabout_shield_matches_reference(self):
        network = RoadNetwork.create_roundabout_scenario(
            car_count=2, road_length=12, max_vel=2
        )
        system = AbstractRoundaboutSystem(network)
        safety = roundabout_safety(min_dist=2)
        with tempfile.TemporaryDirectory(prefix="imashielding-test-") as cache_dir:
            shield = compute(system, safety, cache_dir=cache_dir)
        assert_matches_reference(self, system, safety, shield)
        for init in system.get_init_states():
            self.assertTrue(shield.get_safe_actions(init))


class RoadSafetyAttributionTest(unittest.TestCase):
    """Direct tests of the release-vs-invariant attribution in road_safety."""

    def setUp(self):
        # Only the road topology and lengths matter for these hand-built states.
        self.network = RoadNetwork(
            roads=(
                RoadSegment(0, length=20, end_left=0, end_right=0, start=0),
                RoadSegment(1, length=20, end_left=0, end_right=0, start=0),
            ),
            cars=(
                CarDef(0, 0, 0),
                CarDef(1, 1, 0),
            ),
            max_vel=2,
            min_vel=0,
            accel_actions=((ChancedOutcomeAccel(1.0, 0),),),
        )
        self.safety = road_safety(self.network, min_dist=3, merge_space=8)

    def test_single_car_is_safe(self):
        self.assertIs(
            self.safety(RoadNetworkState((car(0, road=0, pos=5),))),
            StateSafety.SAFE,
        )

    def test_ego_tailgating_is_unsafe(self):
        # cars[0] (the ego) is behind cars[1] with too small a gap.
        state = RoadNetworkState((car(0, road=0, pos=5), car(1, road=0, pos=6)))
        self.assertIs(self.safety(state), StateSafety.UNSAFE)

    def test_another_car_tailgating_the_ego_is_a_release(self):
        # cars[1] is behind the ego (cars[0]) with too small a gap.  The ego is
        # not at fault, so the property is released instead of violated.
        state = RoadNetworkState((car(0, road=0, pos=10), car(1, road=0, pos=9)))
        self.assertIs(self.safety(state), StateSafety.SAFE_RELEASE)

    def test_ego_overtaking_a_slower_car_is_unsafe(self):
        # The ego is physically ahead of the other car (the abstraction wraps
        # that car to "in front") and is still faster: it just overtook.
        state = RoadNetworkState(
            (car(0, road=0, pos=10, vel=2), car(1, road=0, pos=9, vel=1))
        )
        self.assertIs(self.safety(state), StateSafety.UNSAFE)

    def test_ego_being_overtaken_is_a_release(self):
        # The other car moved in front of the ego and is faster: it is the
        # overtaker and therefore the one at fault.
        state = RoadNetworkState(
            (car(0, road=0, pos=9, vel=1), car(1, road=0, pos=10, vel=2))
        )
        self.assertIs(self.safety(state), StateSafety.SAFE_RELEASE)

    def test_another_car_rear_ending_the_ego_is_a_release(self):
        # Same cell, but the other car is the faster one.  This must not fall
        # through to the following-distance rule and blame the ego.
        state = RoadNetworkState(
            (car(0, road=0, pos=5, vel=1), car(1, road=0, pos=5, vel=2))
        )
        self.assertIs(self.safety(state), StateSafety.SAFE_RELEASE)

    def test_ego_rear_ending_another_car_is_unsafe(self):
        state = RoadNetworkState(
            (car(0, road=0, pos=5, vel=2), car(1, road=0, pos=5, vel=1))
        )
        self.assertIs(self.safety(state), StateSafety.UNSAFE)

    def test_merging_ego_must_leave_space_behind(self):
        # The ego recently merged in front of another car and left less than
        # merge_space: this is the ego's fault.
        state = RoadNetworkState(
            (
                car(0, road=0, pos=10, vel=2, just_merged=True),
                car(1, road=0, pos=5, vel=2),
            )
        )
        self.assertIs(self.safety(state), StateSafety.UNSAFE)

    def test_merging_ego_with_enough_space_is_safe(self):
        state = RoadNetworkState(
            (
                car(0, road=0, pos=20 - 1, vel=1, just_merged=True),
                car(1, road=0, pos=1, vel=1),
            )
        )
        # Gap = 18 cells, well above merge_space.
        self.assertIs(self.safety(state), StateSafety.SAFE)


class CrossRoadMergeTest(unittest.TestCase):
    """Cross-road merge checks (the merging car is on the next segment)."""

    def setUp(self):
        # Road 0 is the priority feeder into road 1.  A car that merges onto
        # road 1 must leave ``merge_space`` behind it.
        self.network = RoadNetwork(
            roads=(
                RoadSegment(0, length=20, end_left=1, end_right=1, start=0),
                RoadSegment(1, length=20, end_left=0, end_right=0, start=0),
            ),
            cars=(
                CarDef(0, 1, 0),
                CarDef(1, 0, 0),
            ),
            max_vel=2,
            min_vel=0,
            accel_actions=((ChancedOutcomeAccel(1.0, 0),),),
        )
        self.safety = road_safety(self.network, min_dist=3, merge_space=8)

    def test_merging_ego_too_close_to_priority_car_is_unsafe(self):
        # Ego (id 0) just merged onto road 1 at pos 0; the priority car (id 1)
        # is about to reach the merge point.  gap = 20 - 19 + 0 = 1.
        state = RoadNetworkState(
            (
                car(0, road=1, pos=0, vel=1, just_merged=True),
                car(1, road=0, pos=19, vel=1),
            )
        )
        self.assertIs(self.safety(state), StateSafety.UNSAFE)

    def test_merging_ego_with_enough_space_is_safe(self):
        state = RoadNetworkState(
            (
                car(0, road=1, pos=0, vel=1, just_merged=True),
                car(1, road=0, pos=10, vel=1),
            )
        )
        # gap = 20 - 10 + 0 = 10 >= merge_space.
        self.assertIs(self.safety(state), StateSafety.SAFE)


class RoadNetworkNormalizationTest(unittest.TestCase):
    def test_ego_stays_first_when_cars_share_a_cell(self):
        network = RoadNetwork.create_crossing_roads_scenario(car_count=2)
        system = AbstractRoadNetworkSystem(network)
        state = RoadNetworkState(
            (car(0, road=0, pos=3, vel=1), car(1, road=0, pos=3, vel=2))
        )
        # From car 1's POV the ego must be re-indexed to 0 even though car 0
        # has the smaller original id.
        view = system.normalize(state, 1)
        self.assertEqual(view.cars[0].vel, 2)
        self.assertEqual(view.cars[1].vel, 1)


class CrossingRoadsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = RoadNetwork.create_crossing_roads_scenario(
            car_count=2, road_length_circ=12, max_vel=2
        )
        cls.system = AbstractRoadNetworkSystem(cls.network)
        cls.safety = road_safety(cls.network, min_dist=2, merge_space=2)
        cls._cache_dir = tempfile.TemporaryDirectory(prefix="imashielding-test-")
        cls.addClassCleanup(cls._cache_dir.cleanup)
        cls.shield = compute(cls.system, cls.safety, cache_dir=cls._cache_dir.name)

    def test_crossing_shield_matches_reference(self):
        # ``self.safety`` would bind the closure as a method; use the class
        # attribute to get the plain function object.
        assert_matches_reference(self, self.system, type(self).safety, self.shield)

    def test_crossing_exercises_the_release_property(self):
        # This is the whole point of the crossing-roads scenario: unlike the
        # single roundabout, it must contain non-empty release states.
        self.assertGreater(self.shield.metadata["release_states"], 0)

    def test_every_agents_initial_point_of_view_is_covered(self):
        init = self.network.get_init_state()
        for agent in range(len(self.network.cars)):
            pov = self.system.normalize(init, agent)
            with self.subTest(agent=agent):
                self.assertTrue(
                    self.shield.contains(pov),
                    msg=f"initial POV of agent {agent} is missing from the shield",
                )
                self.assertTrue(self.shield.get_safe_actions(pov))

    def test_normalizing_shields_use_the_same_graph(self):
        init = self.network.get_init_state()
        for agent in range(len(self.network.cars)):
            with self.subTest(agent=agent):
                normalizing = self.shield.with_normalizer(
                    lambda state, a=agent: self.system.normalize(state, a)
                )
                self.assertEqual(
                    normalizing.get_safe_actions(init),
                    self.shield.get_safe_actions(self.system.normalize(init, agent)),
                )
                self.assertTrue(normalizing.contains(init))


class DoubleRoundaboutTest(unittest.TestCase):
    def test_double_roundabout_shield_matches_reference(self):
        network = RoadNetwork.create_double_roundabout_scenario(
            car_count=2, road_per_roundabout=8, max_vel=2
        )
        system = AbstractRoadNetworkSystem(network)
        safety = road_safety(network, min_dist=2, merge_space=2)
        with tempfile.TemporaryDirectory(prefix="imashielding-test-") as cache_dir:
            shield = compute(system, safety, cache_dir=cache_dir)
        assert_matches_reference(self, system, safety, shield)
        for init in system.get_init_states():
            self.assertTrue(shield.get_safe_actions(init))


if __name__ == "__main__":
    unittest.main()
