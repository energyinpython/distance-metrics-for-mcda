# distance-metrics-for-mcda

Python 3 library for Multi-Criteria Decision Analysis based on distance metrics. The documentation is provided [here](https://distance-metrics-for-mcda.readthedocs.io/en/latest/)

# Installation

```
pip install distance-metrics-mcda
```

# Methods

This is Python 3 library providing package `distance_metrics_mcda` that includes metrics that can measure alternatives distance from 
reference solutions in multi-criteria decision analysis. This library contains module `weighting_methods` with the following distance metrics:

- Euclidean distance `euclidean`
- Manhattan (Taxicab) distance `manhattan`
- Hausdorff distance `hausdorff`
- Correlation distance `correlation`
- Chebyshev distance `chebyshev`
- Standardized euclidean distance `std_euclidean`
- Cosine distance `cosine`
- Cosine similarity measure `csm`
- Squared Euclidean distance `squared_euclidean`
- Sorensen or Bray-Curtis distance `bray_curtis`
- Canberra distance `canberra`
- Lorentzian distance `lorentzian`
- Jaccard distance `jaccard`
- Dice distance `dice`
- Bhattacharyya distance `bhattacharyya`
- Hellinger distance `hellinger`
- Matusita distance `matusita`
- Squared-chord distance `squared_chord`
- Pearson chi-square distance `pearson_chi_square`
- Squared chi-square distance `squared_chi_square`

The library also provides other methods necessary for multi-criteria decision analysis, which are as follows: The TOPSIS method for 
multi-criteria decision analysis TOPSIS in module `mcda_methods`. The TOPSIS method is based on measuring the distance of alternatives from 
Positive Ideal Solution and Negative Ideal Solution using `distance_metrics` mentioned above.

Normalization techniques:

- Linear `linear_normalization`
- Minimum-Maximum `minmax_normalization`
- Maximum `max_normalization`
- Sum `sum_normalization`
- Vector `vector_normalization`

Correlation coefficients:

- Spearman rank correlation coefficient rs `spearman`
- Weighted Spearman rank correlation coefficient rw `weighted_spearman`
- Pearson coefficent `pearson_coeff`

Objective weighting methods:

- Entropy weighting method `entropy_weighting`
- CRITIC weighting method `critic_weighting`

Example of usage of `distance-metrics-mcda` are provided on [GitHub](https://github.com/energyinpython/distance-metrics-for-mcda) in [examples](https://github.com/energyinpython/distance-metrics-for-mcda/tree/main/examples)