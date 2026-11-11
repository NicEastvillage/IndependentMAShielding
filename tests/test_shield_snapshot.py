"""Opt-in snapshot test for the full default crossing-roads scenario.

The default 2-car crossing scenario currently takes ~55 seconds to synthesise
and produces a shield with ~100k states.  It is the scenario the performance work
targets, so it is worth checking that optimisation does not change its result.
Because it is slow, it is skipped unless ``IMASHIELDING_SLOW_TESTS=1`` is set.

The fingerprint is a hash of ``repr(state) -> sorted safe actions`` for every
state in the synthesized shield.  It is independent of state ordering and of
the internal representation, but it does change if the synthesizer starts or
stops discovering states.

Run the slow tests with::

    $env:IMASHIELDING_SLOW_TESTS = "1"
    python -m unittest discover -s tests
"""

import contextlib
import hashlib
import io
import os
import tempfile
import unittest

from imashielding.roads import RoadNetwork
from imashielding.roads_reduction import AbstractRoadNetworkSystem, road_safety
from imashielding.shield import compute_shield

# Fingerprint after fixing the fault attribution in ``road_safety`` (2026-11-10)
DEFAULT_CROSSING_FINGERPRINT = (
    "2a0711e3d8cca2751e0cac401256831934b06a4b1c429eb3629833d3d231afd4"
)
DEFAULT_CROSSING_STATES = 104_871
DEFAULT_CROSSING_RELEASE_STATES = 1_797


def shield_fingerprint(shield):
    lines = []
    for state in shield.iter_states():
        if shield.is_released(state):
            actions = ("RELEASED",)
        else:
            actions = tuple(sorted(shield.get_safe_actions(state)))
        lines.append(f"{state!r}::{actions!r}")
    lines.sort()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@unittest.skipUnless(
    os.environ.get("IMASHIELDING_SLOW_TESTS") == "1",
    "slow test: set IMASHIELDING_SLOW_TESTS=1 to run it",
)
class DefaultCrossingSnapshotTest(unittest.TestCase):
    def test_default_crossing_shield_is_unchanged(self):
        network = RoadNetwork.create_crossing_roads_scenario(car_count=2)
        system = AbstractRoadNetworkSystem(network)
        safety = road_safety(network)

        with tempfile.TemporaryDirectory(prefix="imashielding-slow-") as cache_dir:
            with contextlib.redirect_stdout(io.StringIO()):
                shield = compute_shield(
                    system, safety, cache_dir=cache_dir, force_recompute=True
                )

        fingerprint = shield_fingerprint(shield)
        state_count = shield.metadata["safe_states"] + shield.metadata["unsafe_states"]
        summary = (
            f"fingerprint={fingerprint}\n"
            f"state_count={state_count}\n"
            f"release_states={shield.metadata['release_states']}\n"
            f"graph_nodes={shield.metadata['graph_nodes']}"
        )
        self.assertEqual(
            fingerprint,
            DEFAULT_CROSSING_FINGERPRINT,
            msg="the default crossing shield changed:\n" + summary,
        )
        # These are diagnostics as much as assertions; if only the internal
        # representation changes, update them together with the fingerprint
        # after reviewing the diff.
        self.assertEqual(state_count, DEFAULT_CROSSING_STATES, msg=summary)
        self.assertEqual(
            shield.metadata["release_states"],
            DEFAULT_CROSSING_RELEASE_STATES,
            msg=summary,
        )


if __name__ == "__main__":
    unittest.main()
