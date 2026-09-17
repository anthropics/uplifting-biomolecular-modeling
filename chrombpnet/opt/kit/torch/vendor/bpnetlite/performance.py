# performance.py
# Authors: Alex Tseng <amtseng@stanford.edu>
#          Jacob Schreiber <jmschreiber91@gmail.com>

"""
This module contains performance measures that are used to evaluate the
model but are not explicitly used as losses for optimization.

IMPORTANT: MANY OF THESE FUNCTIONS ASSUME THE INPUTS TO BE PREDICTED LOG
PROBABILITIES AND TRUE COUNTS. THE FIRST ARGUMENT MUST BE IN LOG SPACE
AND THE SECOND ARGUMENT MUST BE IN COUNT SPACE FOR THESE FUNCTIONS.
"""

import torch

from .losses import MNLLLoss
from .losses import log1pMSELoss


def smooth_gaussian1d(x, kernel_sigma, kernel_width):
	"""Smooth a signal along the sequence length axis.

	This function is a replacement for the scipy.ndimage.gaussian1d
	function that works on PyTorch tensors. It applies a Gaussian kernel
	to each position which is equivalent to applying a convolution across
	the sequence with weights equal to that of a Gaussian distribution.
	Each sequence, and each channel within the sequence, is smoothed
	independently.


	Parameters
	----------
	x: torch.tensor, shape=(n_sequences, n_channels, seq_len)
		A tensor to smooth along the last axis. n_channels must be at
		least 1.

	kernel_sigma: float
		The standard deviation of the Gaussian to be applied.

	kernel_width: int
		The width of the kernel to be applied.


	Returns
	-------
	x_smooth: torch.tensor, shape=(n_sequences, n_channels, seq_len)
		The smoothed tensor.
	"""

	meshgrid = torch.arange(kernel_width, dtype=torch.float32,
		device=x.device)

	mean = (kernel_width - 1.) / 2.
	kernel = torch.exp(-0.5 * ((meshgrid - mean) / kernel_sigma) ** 2.0)
	kernel = kernel / torch.sum(kernel)
	kernel = kernel.reshape(1, 1, kernel_width).repeat(x.shape[1], 1, 1)
	return torch.nn.functional.conv1d(x, weight=kernel, groups=x.shape[1], 
		padding='same')


def batched_smoothed_function(logps, true_counts, f, smooth_predictions=False, 
	smooth_true=False, kernel_sigma=7, kernel_width=81, 
	exponentiate_logps=False, batch_size=200):
	"""Batch a calculation with optional smoothing.

	Given a set of predicted and true values, apply some function to them in
	a batched manner and store the results. Optionally, either the true values
	or the predicted ones can be smoothed.


	Parameters
	----------
	logps: torch.tensor
		A tensor of the predicted log probability values.

	true_counts: torch.tensor
		A tensor of the true values, usually integer counts.

	f: function
		A function to be applied to the predicted and true values.

	smooth_predictions: bool, optional
		Whether to apply a Gaussian filter to the predictions. Default is 
		False.

	smooth_true: bool, optional
		Whether to apply a Gaussian filter to the true values. Default is
		False.

	kernel_sigma: float, optional
		The standard deviation of the Gaussian to be applied. Default is 7.

	kernel_width: int, optional
		The width of the kernel to be applied. Default is 81.

	exponentiate_logps: bool, optional
		Whether to exponentiate each batch of log probabilities. Default is
		False.

	batch_size: int, optional
		The number of examples in each batch to evaluate at a time. Default
		is 200.


	Returns
	-------
	results: torch.tensor
		The results of applying the function to the tensor.
	"""

	n = logps.shape[0]
	results = torch.empty(*logps.shape[:2])

	for start in range(0, n, batch_size):
		end = start + batch_size

		logps_ = logps[start:end]
		true_counts_ = true_counts[start:end]

		if smooth_predictions:
			logps_ = torch.exp(logps_)
			logps_ = smooth_gaussian1d(logps_, kernel_sigma, kernel_width)

			if exponentiate_logps == False:
				logps_ = torch.log(logps_)
		else:
			if exponentiate_logps:
				logps_ = torch.exp(logps_)

		if smooth_true:
			true_counts_ = smooth_gaussian1d(true_counts_, kernel_sigma, kernel_width)

		results[start:end] = f(logps_, true_counts_) 

	return results


def _kl_divergence(probs1, probs2):
	"""The KL-divergence between two tensors across the last axis.
	
	Computes the KL divergence in the last dimension of `probs1` and `probs2`
	as KL(P1 || P2). `probs1` and `probs2` must be the same shape. For example,
	if they are both A x B x L arrays, then the KL divergence of corresponding
	L-arrays will be computed and returned in an A x B array. Does not
	renormalize the arrays. If probs2[i] is 0, that value contributes 0.


	Parameters
	----------
	logps: torch.Tensor
		A tensor of probabilities. This must not be log probabilities.

	true_counts: torch.Tensor
		A tensor of probabilities. This must not be log probabilities.


	Returns
	-------
	kl: torch.Tensor
		The KL divergence for each element.
	"""

	idxs = ((probs1 != 0) & (probs2 != 0))
	quot_ = torch.divide(probs1, probs2)

	quot = torch.ones_like(probs1)
	quot[idxs] = quot_[idxs]
	return torch.sum(probs1 * torch.log(quot), dim=-1)


def jensen_shannon_distance(logps, true_counts):
	"""The Jensen-Shannon distance between two tensors across the last axis.

	Computes the Jensen-Shannon distance in the last dimension of `logps` and
	`true_counts`. These two tensors must be the same shape. For example, if they
	are both A x B x L arrays, then the KL divergence of corresponding L-arrays
	will be computed and returned in an A x B array. This will renormalize the
	arrays so that each subarray sums to 1. If the sum of a subarray is 0, then
	the resulting JSD will be NaN.


	Parameters
	----------
	logps: torch.Tensor
		A tensor of log probabilities or logits. This must be in log space.

	true_counts: torch.Tensor
		A tensor of true integer counts at each position.


	Returns
	-------
	jsd: torch.Tensor
		The Jensen-Shannon divergence for each element.
	"""
	# Renormalize both distributions, and if the sum is NaN, put NaNs all around

	probs1 = torch.exp(logps)
	probs1_sum = torch.sum(probs1, dim=-1, keepdims=True)
	probs1 = torch.divide(probs1, probs1_sum, out=torch.zeros_like(probs1))

	probs2_sum = torch.sum(true_counts, dim=-1, keepdims=True)
	probs2 = torch.divide(true_counts, probs2_sum, out=torch.zeros_like(true_counts))
	probs2 = probs2.type(probs1.dtype)

	mid = 0.5 * (probs1 + probs2)
	return 0.5 * (_kl_divergence(probs1, mid) + _kl_divergence(probs2, mid))


def pearson_corr(arr1, arr2):
	"""The Pearson correlation between two tensors across the last axis.

	Computes the Pearson correlation in the last dimension of `arr1` and `arr2`.
	`arr1` and `arr2` must be the same shape. For example, if they are both
	A x B x L arrays, then the correlation of corresponding L-arrays will be
	computed and returned in an A x B array.


	Parameters
	----------
	arr1: torch.Tensor
		One of the tensor to correlate.

	arr2: torch.Tensor
		The other tensor to correlation.


	Returns
	-------
	correlation: torch.Tensor
		The correlation for each element where `n` is arr1.shape[-1].
	"""

	mean1 = torch.mean(arr1, axis=-1).unsqueeze(-1)
	mean2 = torch.mean(arr2, axis=-1).unsqueeze(-1)
	dev1, dev2 = arr1 - mean1, arr2 - mean2

	sqdev1, sqdev2 = torch.square(dev1), torch.square(dev2)
	numer = torch.sum(dev1 * dev2, axis=-1)  # Covariance
	var1, var2 = torch.sum(sqdev1, axis=-1), torch.sum(sqdev2, axis=-1)  # Variances
	denom = torch.sqrt(var1 * var2)

	# Divide numerator by denominator, but use 0 where the denominator is 0
	correlation = torch.zeros_like(numer)
	correlation[denom != 0] = numer[denom != 0] / denom[denom != 0]
	return correlation


def spearman_corr(arr1, arr2):
	"""The Spearman correlation between two tensors across the last axis.

	Computes the Spearman correlation in the last dimension of `arr1` and `arr2`.
	`arr1` and `arr2` must be the same shape. For example, if they are both
	A x B x L arrays, then the correlation of corresponding L-arrays will be
	computed and returned in an A x B array.

	A dense ordering is used and ties are broken based on position in the
	tensor.


	Parameters
	----------
	arr1: torch.Tensor
		One of the tensor to correlate. This can be any number of dimensions but the MSE is
		calculated across the last dimension.

	arr2: torch.Tensor
		The other tensor to correlation.


	Returns
	-------
	correlation: torch.Tensor
		The correlation for each element.
	"""

	ranks1 = arr1.argsort().argsort().type(torch.float32)
	ranks2 = arr2.argsort().argsort().type(torch.float32)
	return pearson_corr(ranks1, ranks2)


def mean_squared_error(arr1, arr2):
	"""The mean squared error between two tensors averaged along the last axis.

	Computes the element-wise squared error between two tensors and averages
	these across the last dimension. `arr1` and `arr2` must be the same shape. 
	For example, if they are both A x B x L arrays, then the MSE of 
	corresponding L-arrays will be computed and returned in an A x B array.


	Parameters
	----------
	arr1: torch.Tensor
		A tensor of values.

	arr2: torch.Tensor
		Another tensor of values with the same shape as arr1.


	Returns
	-------
	mse: torch.Tensor
		The L2 distance between two tensors.
	"""

	return torch.mean(torch.square(arr1 - arr2), axis=-1)


def calculate_performance_measures(logps, true_counts, pred_log_counts,
	kernel_sigma=7, kernel_width=81, smooth_true=False, 
	smooth_predictions=False, measures=None):
	"""Calculates a set of performance measures given true and observed data.

	This function will take in observed readouts, predicted profiles, and
	predicted counts, and calculate a series of specified performance measures
	on them. Each performance measure could be calculated individually using
	its function, but this function provides a wrapper around running any
	number of them. The measures one can choose are:

		Profile Performance Measures
		- "profile_mnll": the multinomial log-likelihood of the observed profile given
			the predicted logits
		- "profile_jsd": the Jensen-Shannon divergence between the observed profile and
			the predicted probabilities
		- "profile_pearson": the Pearson correlation between the observed profile and
			the predicted probabilities
		- "profile_spearman": the Spearman correlation between the observed profiles
			and the predicted probabilities

		Count Performance Measures
		- "count_pearson": the Pearson correlation between the observed log counts and
			the predicted log counts
		- "count_spearman": the Spearman correlation between the observed log counts
			and the predicted log counts
		- "count_mse": the mean-squared error between the observed log counts and the
			predicted log counts.

	Optionally, one can choose to smooth the *observed* data before calculating
	the profile correlations and JSD. It is important to note that this smoothing
	is not being done on the predictions, but on the observed bp-resolution
	counts, with the reasoning being that these counts are sparse due to their
	bp resolution nature. The smoothing is done according to a Gaussian with a
	sigma and kernel width as specified.


	Parameters
	----------
	logps: torch.Tensor, shape=(n, n_strands, length)
		The predicted logits or log probabilities for each basepir for each strand. 
		If the predictions are unstranded, this dimension must be 1.

	true_counts: torch.Tensor, shape=(n, n_strands, length)
		The integer counts of the number of reads per basepair in the observed data.
	
	pred_log_counts: torch.Tensor, shape=(n, n_outputs)
		The predicted log counts for each example.

	kernel_sigma: int, optional
		If smoothing the observed profile, the sigma to use in the Gaussian
		smoothing. Default is 7.

	kernel_width: int, optional
		If smoothing the observed profile, the kernel width to use in the Gaussian
		smoothing. Default is 81.

	smooth_true: bool, optional
		Whether to smooth the observed data using a Gassian kernel. Default is False.

	smooth_predictions: bool, optional
		Whether to smooth the predicted values using a Gaussian kernel. Default is
		False.

	measures: None or list, optional
		If a list of strings, each string should correspond to a performance measure
		to calculate. If None, calculate all performance measures.


	Returns
	-------
	measures_: dict of torch.Tensors
		A dictionary where the keys are the names of performance measures and the
		values are tensors containing the values. Each profile performance measure
		will have the shape (n, 1) and each count performance measure will have
		the shape (1,).
	"""

	measures_ = {}
	logps = torch.nn.functional.log_softmax(logps, dim=-1)

	# Profile Performance Meausres
	if measures is None or 'profile_mnll' in measures: 
		measures_['profile_mnll'] = batched_smoothed_function(logps=logps, 
			true_counts=true_counts, f=MNLLLoss, 
			smooth_predictions=smooth_predictions, smooth_true=False, 
			kernel_sigma=kernel_sigma, kernel_width=kernel_width)

	if measures is None or 'profile_jsd' in measures: 
		measures_['profile_jsd'] = batched_smoothed_function(logps=logps, 
			true_counts=true_counts, f=jensen_shannon_distance, 
			smooth_predictions=smooth_predictions, smooth_true=smooth_true, 
			kernel_sigma=kernel_sigma, kernel_width=kernel_width)

	if measures is None or 'profile_pearson' in measures:
		measures_['profile_pearson'] = batched_smoothed_function(logps=logps, 
			true_counts=true_counts, f=pearson_corr, 
			smooth_predictions=smooth_predictions, smooth_true=smooth_true, 
			exponentiate_logps=True, kernel_sigma=kernel_sigma, 
			kernel_width=kernel_width)

	if measures is None or 'profile_spearman' in measures:
		measures_['profile_spearman'] = batched_smoothed_function(logps=logps, 
			true_counts=true_counts, f=spearman_corr, 
			smooth_predictions=smooth_predictions, smooth_true=smooth_true, 
			exponentiate_logps=True, kernel_sigma=kernel_sigma, 
			kernel_width=kernel_width)


	# Count Performance Measures
	true_counts = true_counts.sum(dim=(-1, -2)).unsqueeze(-1)
	true_log_counts = torch.log(true_counts + 1)

	if measures is None or 'count_pearson' in measures:
		measures_['count_pearson'] = pearson_corr(pred_log_counts.T, 
			true_log_counts.T)

	if measures is None or 'count_spearman' in measures:
		measures_['count_spearman'] = spearman_corr(pred_log_counts.T, 
			true_log_counts.T)

	if measures is None or 'count_mse' in measures:
		measures_['count_mse'] = mean_squared_error(pred_log_counts.T, 
			true_log_counts.T)

	return measures_
