Usage
======

.. _installation:

Installation
-------------

To use distance_metrics_mcda, first install it using pip:

.. code-block:: python

	pip install distance_metrics_mcda


Usage examples
----------------------

The TOPSIS method
___________________

.. code-block:: python

	import numpy as np
	from distance_metrics_mcda.mcda_methods import TOPSIS
	from distance_metrics_mcda import normalizations as norms
	from distance_metrics_mcda import distance_metrics as dists
	from distance_metrics_mcda.additions import rank_preferences

	# provide decision matrix in array numpy.darray

	matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
	[256, 8, 32, 1.0, 1.8, 6919.99],
	[256, 8, 53, 1.6, 1.9, 8400],
	[256, 8, 41, 1.0, 1.75, 6808.9],
	[512, 8, 35, 1.6, 1.7, 8479.99],
	[256, 4, 35, 1.6, 1.7, 7499.99]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.

	weights = np.array([0.405, 0.221, 0.134, 0.199, 0.007, 0.034])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.

	types = np.array([1, 1, 1, 1, -1, -1])

	# Create the TOPSIS method object providing normalization method and distance metric.

	topsis = TOPSIS(normalization_method = norms.minmax_normalization, distance_metric = dists.euclidean)

	# Calculate the TOPSIS preference values of alternatives

	pref = topsis(matrix, weights, types)

	# Generate ranking of alternatives by sorting alternatives descendingly according to the TOPSIS algorithm (reverse = True means sorting in descending order) according to preference values

	rank = rank_preferences(pref, reverse = True)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [0.4242 0.3217 0.4453 0.3353 0.8076 0.2971]
	Ranking:  [3 5 2 4 1 6]


Correlation coefficents
__________________________

Spearman correlation coefficient

.. code-block:: python

	import numpy as np
	from distance_metrics_mcda import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `spearman` coefficient
	coeff = corrs.spearman(R, Q)
	print('Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Spearman coeff:  0.9

	
	
Weighted Spearman correlation coefficient


.. code-block:: python

	import numpy as np
	from distance_metrics_mcda import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `weighted_spearman` coefficient
	coeff = corrs.weighted_spearman(R, Q)
	print('Weighted Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Weighted Spearman coeff:  0.8833
	
	
Pearson correlation coefficient


.. code-block:: python

	import numpy as np
	from distance_metrics_mcda import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `pearson_coeff` coefficient
	coeff = corrs.pearson_coeff(R, Q)
	print('Pearson coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Pearson coeff:  0.9
	
	
	
Methods for criteria weights determination
___________________________________________

Entropy weighting method

		
.. code-block:: python

	import numpy as np
	from distance_metrics_mcda import weighting_methods as mcda_weights

	matrix = np.array([[30, 30, 38, 29],
	[19, 54, 86, 29],
	[19, 15, 85, 28.9],
	[68, 70, 60, 29]])

	weights = mcda_weights.entropy_weighting(matrix)

	print('Entropy weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	Entropy weights:  [0.463  0.3992 0.1378 0.    ]
	

CRITIC weighting method
		
.. code-block:: python

	import numpy as np
	from distance_metrics_mcda import weighting_methods as mcda_weights

	matrix = np.array([[5000, 3, 3, 4, 3, 2],
	[680, 5, 3, 2, 2, 1],
	[2000, 3, 2, 3, 4, 3],
	[600, 4, 3, 1, 2, 2],
	[800, 2, 4, 3, 3, 4]])

	weights = mcda_weights.critic_weighting(matrix)

	print('CRITIC weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	CRITIC weights:  [0.157  0.2495 0.1677 0.1211 0.1541 0.1506]
	
	
Distance metrics
_________________

Here are two examples of using distance metrics for Euclidean distance ``euclidean`` and Manhattan distance ``manhattan``. Usage of other distance metrics
provided in module ``distance metrics`` is analogous.


Euclidean distance


.. code-block:: python

	import numpy as np
	from distance_metrics_mcda import distance_metrics as dists

	A = np.array([0.165, 0.113, 0.015, 0.019])
	B = np.array([0.227, 0.161, 0.053, 0.130])

	dist = dists.euclidean(A, B)
	print('Distance: ', np.round(dist, 4))
	
Output

.. code-block:: console

	Distance:  0.1411
	
	
Manhattan distance


.. code-block:: python

	import numpy as np
	from distance_metrics_mcda import distance_metrics as dists

	A = np.array([0.165, 0.113, 0.015, 0.019])
	B = np.array([0.227, 0.161, 0.053, 0.130])

	dist = dists.manhattan(A, B)
	print('Distance: ', np.round(dist, 4))
	
Output

.. code-block:: console

	Distance:  0.259
	
	
Normalization methods
______________________

Here is an example of vector normalization usage. Other normalizations provided in module ``normalizations``, namely ``minmax_normalization``, ``max_normalization``,
``sum_normalization``, ``linear_normalization``, ``multimoora_normalization`` are used in analogous way.


Vector normalization


.. code-block:: python

	import numpy as np
	from distance_metrics_mcda import normalizations as norms

	matrix = np.array([[8, 7, 2, 1],
	[5, 3, 7, 5],
	[7, 5, 6, 4],
	[9, 9, 7, 3],
	[11, 10, 3, 7],
	[6, 9, 5, 4]])

	types = np.array([1, 1, 1, 1])

	norm_matrix = norms.vector_normalization(matrix, types)
	print('Normalized matrix: ', np.round(norm_matrix, 4))
	
Output

.. code-block:: console

	Normalized matrix:  [[0.4126 0.3769 0.1525 0.0928]
	 [0.2579 0.1615 0.5337 0.4642]
	 [0.361  0.2692 0.4575 0.3714]
	 [0.4641 0.4845 0.5337 0.2785]
	 [0.5673 0.5384 0.2287 0.6499]
	 [0.3094 0.4845 0.3812 0.3714]]

