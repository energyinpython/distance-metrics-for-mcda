import unittest
import numpy as np
from distance_metrics_mcda import normalizations as norms
from distance_metrics_mcda import weighting_methods as mcda_weights


# Test for CRITIC weighting
class Test_CRITIC(unittest.TestCase):

    def test_critic(self):
        """Test based on paper Tuş, A., & Aytaç Adalı, E. (2019). The new combination with CRITIC and WASPAS methods 
        for the time and attendance software selection problem. Opsearch, 56(2), 528-538."""

        matrix = np.array([[5000, 3, 3, 4, 3, 2],
        [680, 5, 3, 2, 2, 1],
        [2000, 3, 2, 3, 4, 3],
        [600, 4, 3, 1, 2, 2],
        [800, 2, 4, 3, 3, 4]])

        types = np.array([-1, 1, 1, 1, 1, 1])

        test_result = mcda_weights.critic_weighting(matrix)
        real_result = np.array([0.157, 0.249, 0.168, 0.121, 0.154, 0.151])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for Entropy weighting
class Test_Entropy(unittest.TestCase):

    def test_Entropy(self):
        """Test based on paper Xu, X. (2004). A note on the subjective and objective integrated approach to 
        determine attribute weights. European Journal of Operational Research, 156(2), 
        530-532."""

        matrix = np.array([[30, 30, 38, 29],
        [19, 54, 86, 29],
        [19, 15, 85, 28.9],
        [68, 70, 60, 29]])

        types = np.array([1, 1, 1, 1])

        test_result = mcda_weights.entropy_weighting(matrix)
        real_result = np.array([0.4630, 0.3992, 0.1378, 0.0000])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))

    def test_Entropy2(self):
        """Test based on paper Zavadskas, E. K., & Podvezko, V. (2016). Integrated determination of objective 
        criteria weights in MCDM. International Journal of Information Technology & Decision 
        Making, 15(02), 267-283."""

        matrix = np.array([[3.0, 100, 10, 7],
        [2.5, 80, 8, 5],
        [1.8, 50, 20, 11],
        [2.2, 70, 12, 9]])

        types = np.array([-1, 1, -1, 1])

        test_result = mcda_weights.entropy_weighting(matrix)
        real_result = np.array([0.1146, 0.1981, 0.4185, 0.2689])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))
        


def main():
    test_critic = Test_CRITIC()
    test_critic.test_critic()

    test_entropy = Test_Entropy()
    test_entropy.test_Entropy()
    test_entropy.test_Entropy2()

    

if __name__ == '__main__':
    main()

