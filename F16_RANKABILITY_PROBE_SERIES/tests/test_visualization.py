#!/usr/bin/env python3
"""
Unit test suite for F16 candidate geometry & selection diagnostics visualizer.
Validates dynamic computation, metric integrity, coincidence logic, and dual-panel rendering.
"""

import os
import sys
import unittest
import tempfile
import numpy as np

# Set MPLCONFIGDIR before importing matplotlib
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
workspace_root = os.path.dirname(parent_dir)
cache_mpl_dir = os.path.join(workspace_root, ".cache", "mpl_config")
os.makedirs(cache_mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_mpl_dir
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from visualize_candidate_geometry import compute_sample_diagnostics, draw_dual_panel


class TestVisualizationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache_path = os.path.join(parent_dir, "results", "cache", "val_seed1.npz")
        assert os.path.exists(cache_path), f"Cache not found: {cache_path}"
        cls.cache = np.load(cache_path)
        cls.diagnostics = compute_sample_diagnostics(cls.cache)

    def test_diagnostics_dimensions_and_keys(self):
        """Verify dynamic calculation matches expected shapes from cache."""
        N = len(self.cache['sample_id'])
        self.assertEqual(self.diagnostics["N"], N)
        self.assertEqual(self.diagnostics["K"], 20)
        self.assertEqual(len(self.diagnostics["regret"]), N)
        self.assertEqual(len(self.diagnostics["endpoint_spread"]), N)
        self.assertEqual(self.diagnostics["normalized_scores"].shape, (N, 20))

    def test_regret_and_minade_identity(self):
        """Verify regret is strictly non-negative and matches ADE differences."""
        ADE_final = self.cache['ADE_final']
        sel_idx = self.cache['native_selected_index']
        oracle_idx = self.cache['oracle_final_index']

        for i in range(100):
            expected_regret = ADE_final[i, sel_idx[i]] - ADE_final[i, oracle_idx[i]]
            self.assertAlmostEqual(float(self.diagnostics["regret"][i]), float(expected_regret), places=5)
            self.assertGreaterEqual(float(self.diagnostics["regret"][i]), -1e-6)

    def test_endpoint_spread_formula(self):
        """Verify average pairwise distance on candidate endpoints."""
        candidates = self.cache['candidate_final'][:5]  # [5, 20, 12, 2]
        endpoints = candidates[:, :, -1, :]             # [5, 20, 2]

        for i in range(5):
            dists = []
            for j in range(20):
                for k in range(20):
                    if j != k:
                        d = np.linalg.norm(endpoints[i, j] - endpoints[i, k])
                        dists.append(d)
            expected_spread = np.mean(dists)
            self.assertAlmostEqual(float(self.diagnostics["endpoint_spread"][i]), float(expected_spread), places=5)

    def test_single_panel_render(self):
        """Verify end-to-end rendering generates a non-empty image file."""
        test_out_dir = os.path.join(workspace_root, ".cache", "test_viz_out")
        os.makedirs(test_out_dir, exist_ok=True)
        out_png = os.path.join(test_out_dir, "test_sample_0.png")
        if os.path.exists(out_png):
            os.remove(out_png)

        draw_dual_panel(
            sample_idx=0,
            cache_data=self.cache,
            diagnostics=self.diagnostics,
            output_filepath=out_png,
            show_anchors=True,
            dpi=100
        )
        self.assertTrue(os.path.exists(out_png))
        file_size = os.path.getsize(out_png)
        self.assertGreater(file_size, 20000, "Rendered panel image should be at least 20KB")


if __name__ == "__main__":
    unittest.main()
