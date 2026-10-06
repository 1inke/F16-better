import unittest
import numpy as np
from classification import diagnose


class DiagnosticTests(unittest.TestCase):
    def paths(self, errors):
        # Ten-step 1D paths: all but last error chosen to match requested ADE/FDE.
        a = np.zeros((1, len(errors), 10, 2))
        for k, (ade, fde) in enumerate(errors):
            a[0, k, :9, 0] = (10 * ade - fde) / 9
            a[0, k, 9, 0] = fde
        return a

    def cat(self, errors, selected):
        p = self.paths(errors)
        score = np.zeros((1, len(errors))); score[0, selected] = 1
        return diagnose(p, np.zeros((1, 10, 2)), score)[0]

    def test_six_classes(self):
        examples = [([(.1,.1),(.7,.9)],0), ([ (.1,.1),(.7,.9)],1),
                    ([(.2,.8),(.6,.2)],0), ([(.2,.8),(.6,.9)],0),
                    ([(.6,.2),(.7,.9)],0), ([(.6,.8),(.7,.9)],0)]
        for expected, (paths, index) in enumerate(examples):
            self.assertEqual(self.cat(paths, index)["category"][0], expected)

    def test_tie_and_zero_regret(self):
        d = self.cat([(.1,.1),(.1,.1)], 1)
        self.assertFalse(d["selected_exact_ade_slot"][0])
        self.assertTrue(d["selected_ade_optimal_tolerance"][0])
        self.assertAlmostEqual(d["joint_regret"][0],0)

    def test_selection_not_GT(self):
        p = self.paths([(.1,.1),(.8,.9)])
        score = np.array([[0.,2.]])
        d, _, _ = diagnose(p, np.zeros((1,10,2)), score)
        changed, _, _ = diagnose(p, np.full((1,10,2),7.), score)
        self.assertEqual(d["selected"][0],changed["selected"][0])

    def test_regret_decomposition(self):
        d = self.cat([(.1,.2),(.5,.6)],1)
        self.assertAlmostEqual(d["ade_selected"][0],d["minade"][0]+d["ade_regret"][0])
        self.assertAlmostEqual(d["fde_selected"][0],d["minfde"][0]+d["fde_regret"][0])


if __name__ == "__main__": unittest.main()
