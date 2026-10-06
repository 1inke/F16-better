from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_h3_objective_attribution as runner


class H3ObjectiveAttributionTests(unittest.TestCase):
    def test_new_commands_are_validation_only_and_coverage_zero(self) -> None:
        for key in ("d1_o0", "d0_o0"):
            command, _, diversity = runner.command_for(key, runner.ASTRA_DEFAULT)
            self.assertNotIn("--test-data-path", command)
            self.assertNotIn("--test-num-path", command)
            index = command.index("--oracle-coverage-weight")
            self.assertEqual(command[index + 1], "0.00")
            self.assertEqual(diversity, 0.05 if key == "d1_o0" else 0.0)

    def test_only_two_new_cells(self) -> None:
        with self.assertRaises(ValueError):
            runner.command_for("d1_o1", runner.ASTRA_DEFAULT)


if __name__ == "__main__":
    unittest.main()
