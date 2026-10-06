from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_v2_coverage_propagation as analyzer
import run_v2_coverage_propagation as runner


class V2CoveragePropagationTests(unittest.TestCase):
    def test_command_changes_only_h3_source_and_never_receives_test_paths(self) -> None:
        command, env, output = runner.command_for(runner.ASTRA_DEFAULT)
        self.assertNotIn("--test-data-path", command)
        self.assertNotIn("--test-num-path", command)
        self.assertEqual(command[command.index("--oracle-coverage-weight") + 1], "0.20")
        self.assertEqual(command[command.index("--epochs") + 1], "20")
        self.assertEqual(
            env["EXP173_H3_SOURCE_CHECKPOINT"],
            str(runner.H3_OFF_RUN / "best_selected.pth"),
        )
        self.assertEqual(output, runner.NEW_OUTPUT_ROOT)

    def test_propagation_gate_passes_consistent_effect(self) -> None:
        effects = {
            "best": {
                "j_rel": -0.025,
                "minade_rel": -0.023,
                "minfde_rel": -0.020,
                "support_1p0": 0.60,
                "selected_ade_rel": 0.002,
            },
            "final": {
                "j_rel": -0.022,
                "minade_rel": -0.021,
                "minfde_rel": -0.018,
                "support_1p0": 0.55,
                "selected_ade_rel": -0.001,
            },
        }
        self.assertEqual(
            analyzer.classify(effects)[0], "H3_COVERAGE_PROPAGATES_THROUGH_V2"
        )

    def test_propagation_gate_stops_small_effect(self) -> None:
        effects = {
            "best": {
                "j_rel": -0.004,
                "minade_rel": -0.005,
                "minfde_rel": -0.003,
                "support_1p0": 0.10,
                "selected_ade_rel": 0.0,
            },
            "final": {
                "j_rel": 0.002,
                "minade_rel": 0.001,
                "minfde_rel": 0.003,
                "support_1p0": -0.05,
                "selected_ade_rel": 0.0,
            },
        }
        self.assertEqual(
            analyzer.classify(effects)[0], "H3_COVERAGE_GAIN_ABSORBED_BY_V2"
        )


if __name__ == "__main__":
    unittest.main()
