"""Tests for disk-backed shield caching.

Run from ``IndependentMAShielding`` with::

    python -m unittest discover -s tests
"""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from imashielding.roads import RoadNetwork
from imashielding.roads_reduction import AbstractRoundaboutSystem, roundabout_safety
from imashielding.shield import DGShield, compute_shield


class ShieldCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="imashielding-test-")
        self.cache_dir = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"IMASHIELDING_SHIELD_CACHE": str(self.cache_dir)})
        self._env.start()

        self.network = RoadNetwork.create_roundabout_scenario(
            car_count=3, road_length=12, max_vel=3
        )
        self.system = AbstractRoundaboutSystem(self.network)
        self.safety = roundabout_safety(2)
        self.init = self.system.get_init_state()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _compute(self, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            shield = compute_shield(self.system, self.safety, **kwargs)
        return shield, output.getvalue()

    def _cache_files(self):
        return sorted(self.cache_dir.glob("*.shield.pkl.gz"))

    def test_synthesis_is_saved_and_reused(self):
        first, first_output = self._compute()
        self.assertIn("Saved shield to cache", first_output)
        self.assertEqual(len(self._cache_files()), 1)

        second, second_output = self._compute()
        self.assertIn("Loaded shield from cache", second_output)
        self.assertNotIn("Computing shield...", second_output)
        self.assertEqual(first.get_safe_actions(self.init), second.get_safe_actions(self.init))
        self.assertEqual(first.state_count, second.state_count)
        self.assertEqual(first.metadata["graph_nodes"], second.metadata["graph_nodes"])
        self.assertEqual(first.metadata["cache_key"], second.metadata["cache_key"])

    def test_force_recompute_synthesizes_again(self):
        self._compute()
        _, output = self._compute(force_recompute=True)
        self.assertIn("Computing shield...", output)
        self.assertIn("Saved shield to cache", output)

    def test_different_safety_properties_use_different_files(self):
        self._compute()
        # Synthesize the same system with a different property.
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            compute_shield(self.system, roundabout_safety(3))
        self.assertEqual(len(self._cache_files()), 2)

    def test_corrupt_cache_is_ignored_and_replaced(self):
        self._compute()
        for path in self._cache_files():
            path.write_bytes(b"not a shield")

        shield, output = self._compute()
        self.assertIn("Ignoring unusable shield cache", output)
        self.assertIn("Computing shield...", output)
        self.assertTrue(shield.get_safe_actions(self.init))
        self.assertTrue(all(path.stat().st_size > 100 for path in self._cache_files()))

    def test_round_trip_preserves_shield(self):
        first, _ = self._compute()
        path = self._cache_files()[0]
        loaded = DGShield.load(path)
        self.assertEqual(loaded.get_safe_actions(self.init), first.get_safe_actions(self.init))
        self.assertEqual(loaded.state_count, first.state_count)
        self.assertEqual(loaded.metadata["graph_nodes"], first.metadata["graph_nodes"])
        self.assertEqual(loaded.metadata["graph_edges"], first.metadata["graph_edges"])
        self.assertEqual(loaded.metadata["safe_states"], first.metadata["safe_states"])
        self.assertEqual(
            sorted(map(repr, loaded.iter_states())),
            sorted(map(repr, first.iter_states())),
        )


if __name__ == "__main__":
    unittest.main()
