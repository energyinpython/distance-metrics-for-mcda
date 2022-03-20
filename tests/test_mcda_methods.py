from distance_metrics_mcda.mcda_methods import TOPSIS
from distance_metrics_mcda.additions import rank_preferences
from distance_metrics_mcda.normalizations import vector_normalization

import unittest
import numpy as np

# Test for TOPSIS method
class Test_TOPSIS(unittest.TestCase):

    def test_topsis(self):
        """Papathanasiou, J., & Ploskas, N. (2018). Topsis. In Multiple criteria decision aid 
        (pp. 1-30). Springer, Cham."""

        matrix = np.array([[8, 7, 2, 1],
        [5, 3, 7, 5],
        [7, 5, 6, 4],
        [9, 9, 7, 3],
        [11, 10, 3, 7],
        [6, 9, 5, 4]])

        weights = np.array([0.4, 0.3, 0.1, 0.2])

        types = np.array([1, 1, 1, 1])

        method = TOPSIS(normalization_method=vector_normalization)
        test_result = method(matrix, weights, types)
        real_result = np.array([0.387, 0.327, 0.391, 0.615, 0.868, 0.493])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for rank preferences
class Test_Rank_preferences(unittest.TestCase):

    def test_rank_preferences(self):
        """Papathanasiou, J., & Ploskas, N. (2018). Topsis. In Multiple criteria decision aid 
        (pp. 1-30). Springer, Cham."""

        pref = np.array([0.387, 0.327, 0.391, 0.615, 0.868, 0.493])
        test_result =rank_preferences(pref , reverse = False)
        real_result = np.array([5, 6, 4, 2, 1, 3])
        self.assertEqual(list(test_result), list(real_result))


def main():
    test_topsis = Test_TOPSIS()
    test_topsis.test_topsis()

    # test_rank_preferences = Test_Rank_preferences()
    # test_rank_preferences.test_rank_preferences()


if __name__ == '__main__':
    main()