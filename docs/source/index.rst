Welcome to distance_metrics_mcda documentation!
===================================

This is Python 3 library providing package distance_metrics_mcda that includes metrics that can measure alternatives distance from reference 
solutions in multi-criteria decision analysis.

- The TOPSIS method
	
- Distance metrics:

	- `euclidean` (Euclidean distance)
	- `manhattan` (Manhattan distance)
	- `hausdorff` (Hausdorff distance)
	- `correlation` (Correlation distance)
	- `chebyshev` (Chebyshev distance)
	- `std_euclidean` (Standardized Euclidean distance)
	- `cosine` (Cosine distance)
	- `csm` (Cosine similarity measure)
	- `squared_euclidean` (Squared Euclidean distance)
	- `bray_curtis` (Sorensen or Bray-Curtis distance)
	- `canberra` (Canberra distance)
	- `lorentzian` (Lorentzian distance)
	- `jaccard` (Jaccard distance)
	- `dice` (Dice distance)
	- `bhattacharyya` (Bhattacharyya distance)
	- `hellinger` (Hellinger distance)
	- `matusita` (Matusita distance)
	- `squared_chord` (Squared-chord distance)
	- `pearson_chi_square` (Pearson chi square distance)
	- `squared_chi_square` (Sqaured chi square distance)
	
- Correlation coefficients:

	- `spearman` (Spearman rank correlation coefficient)
	- `weighted_spearman` (Weighted Spearman rank correlation coefficient)
	- `pearson_coeff` (Pearson correlation coefficient)
	
- Methods for normalization of decision matrix:

	- `linear_normalization` (Linear normalization)
	- `minmax_normalization` (Minimum-Maximum normalization)
	- `max_normalization` (Maximum normalization)
	- `sum_normalization` (Sum normalization)
	- `vector_normalization` (Vector normalization)
	
- Methods for determination of criteria weights (weighting methods):

	- `entropy_weighting` (Entropy weighting method)
	- `critic_weighting` (CRITIC weighting method)
	
- additions:

	- `rank_preferences` (Method for ordering alternatives according to their preference values obtained with MCDA methods)
	
Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
