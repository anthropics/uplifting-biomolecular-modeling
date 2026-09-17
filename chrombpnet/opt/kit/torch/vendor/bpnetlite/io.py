# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Code adapted from Alex Tseng, Avanti Shrikumar, and Ziga Avsec

import numpy
import torch

from tqdm import tqdm
from tangermeme.io import extract_loci

import time


class PeakNegativeSampler(torch.utils.data.Dataset):
	"""A data generator mimicking the BPNet data loading procedure.
	
	Here, a set of peaks and negatives are separately loaded. These sets can be
	any size. From these sets, batches of given size are sampled that are a
	mixture of peaks and negatives. The peaks are sampled by randomly iterating
	over the entire set such that one epoch means one pass over the entire data
	set. The negatives are sampled by randomly choosing regions from the
	negatives at the given ratio to peaks, without considering whether they have
	been selected before.
	
	Because peaks and negatives are provided as separate tensors, different
	jittering can be used on them. This means that, for instance, jittering can
	be used on peaks but not used on negatives.
	
	In the documentation below, `mj` = max_jitter.
	
	Note that, although the data is passed in as PyTorch tensors, they are saved
	as numpy arrays for faster slicing during training.
	
	
	Parameters
	----------
	peak_sequences: torch.tensor, shape=(n_peaks, 4, in_window+2*mj)
		A tensor of peak sequences that are one-hot encoded. See above for the
		connection between the length here and jitter.

	peak_signals: torch.tensor, shape=(n_peaks, t, out_window+2*mj)
		A tensor of signals to predict, usually base-pair resolution integer counts.
		This should have `t` tasks, which is usually 2 if predicting stranded outputs
		and 1 if predicting unstranded outputs. See above for the connection between
		the length here and jitter.

	peak_controls: torch.tensor, shape=(n, t, out_window+2*mj) or None, optional
		A tenso of the control signal to take as input, usually base-pair counts, for `n`
		examples with `t` strands and output length `out_window`. If
		None, does not return controls.

	negative_sequences: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
		A one-hot encoded tensor of `n` example sequences, each of input 
		length `in_window`. See description above for connection with jitter.

	negative_signals: torch.tensor, shape=(n, t, out_window+2*max_jitter)
		The signals to predict, usually counts, for `n` examples with
		`t` output tasks (usually 2 if stranded, 1 otherwise), each of 
		output length `out_window`. See description above for connection 
		with jitter.

	negative_controls: torch.tensor, shape=(n, t, out_window+2*max_jitter) or None, optional
		The control signal to take as input, usually counts, for `n`
		examples with `t` strands and output length `out_window`. If
		None, does not return controls.
	
	p: torch.tensor or None, shape=(n,) 
		A vector of probabilities that sum to 1 containing the sampling probability
		of each sequence. If None, use a uniform distribution.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 0.

	reverse_complement: bool, optional
		Whether to reverse complement-augment half of the data. Default is False.

	random_state: int or None, optional
		Whether to use a deterministic seed or not.
	"""
	
	def __init__(self, peak_sequences, peak_signals, negative_sequences, 
		negative_signals, peak_controls=None, negative_controls=None, 
		negative_ratio=0.1, in_window=2114, out_window=1000, max_jitter=0, 
		reverse_complement=False, shuffle=True, random_state=None):
		self.peak_sequences = peak_sequences.numpy(force=True)
		self.peak_signals = peak_signals.numpy(force=True)
		self.n_peaks = len(self.peak_sequences)
		
		self.negative_sequences = negative_sequences.numpy(force=True)
		self.negative_signals = negative_signals.numpy(force=True)
		self.n_negatives = len(self.negative_sequences)

		if peak_controls is not None:
			self.peak_controls = peak_controls.numpy(force=True)
			self.negative_controls = negative_controls.numpy(force=True)
		else:
			self.peak_controls = None
			self.negative_controls = None

		self.negative_ratio = negative_ratio
		self.negative_likelihood = 1 / (1 + 1 / negative_ratio)

		self.in_window = in_window
		self.out_window = out_window
		self.max_jitter = max_jitter
		self.reverse_complement = reverse_complement
		self.shuffle = shuffle

		self.random_state = numpy.random.RandomState(random_state)
		self.n_peaks_seen = 0
		self.peak_ordering = None

	def __len__(self):
		return self.n_peaks + int(self.n_peaks * self.negative_ratio)

	def __getitem__(self, idx):
		if idx == 0:
			self.peak_ordering = numpy.arange(self.n_peaks)
			if self.shuffle:
				self.random_state.shuffle(self.peak_ordering)


		if self.random_state.uniform() >= self.negative_likelihood:
			idx = self.peak_ordering[self.n_peaks_seen % self.n_peaks]
			jitter = self.random_state.randint(self.max_jitter*2)
			label = 1
 
			X, y, X_ctl = self.peak_sequences, self.peak_signals, self.peak_controls
			self.n_peaks_seen += 1

		else:
			idx = self.random_state.randint(self.n_negatives)
			jitter = 0
			label = 0

			X, y, X_ctl = (self.negative_sequences, self.negative_signals, 
				self.negative_controls)


		Xi = torch.from_numpy(X[idx][:, jitter:jitter+self.in_window])
		yi = torch.from_numpy(y[idx][:, jitter:jitter+self.out_window])
		if self.peak_controls is not None:
			Xi_ctl = torch.from_numpy(X_ctl[idx][:, jitter:jitter+self.in_window])


		if self.reverse_complement and self.random_state.randint(2) == 1:
			Xi = torch.flip(Xi, [0, 1])
			yi = torch.flip(yi, [0, 1])
			
			if self.peak_controls is not None:
				Xi_ctl = torch.flip(Xi_ctl, [0, 1])


		if self.peak_controls is not None:
			return Xi, Xi_ctl, yi, label
		
		return Xi, yi, label


class DataGenerator(torch.utils.data.Dataset):
	"""A data generator for BPNet inputs.

	This generator takes in an extracted set of sequences, output signals,
	and control signals, and will return a single element with random
	jitter and reverse-complement augmentation applied. Jitter is implemented
	efficiently by taking in data that is wider than the in/out windows by
	two times the maximum jitter and windows are extracted from that.
	Essentially, if an input window is 1000 and the maximum jitter is 128, one
	would pass in data with a length of 1256 and a length 1000 window would be
	extracted starting between position 0 and 256. This  generator must be 
	wrapped by a PyTorch generator object.

	Parameters
	----------
	sequences: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
		A one-hot encoded tensor of `n` example sequences, each of input 
		length `in_window`. See description above for connection with jitter.

	signals: torch.tensor, shape=(n, t, out_window+2*max_jitter)
		The signals to predict, usually counts, for `n` examples with
		`t` output tasks (usually 2 if stranded, 1 otherwise), each of 
		output length `out_window`. See description above for connection 
		with jitter.

	controls: torch.tensor, shape=(n, t, out_window+2*max_jitter) or None, optional
		The control signal to take as input, usually counts, for `n`
		examples with `t` strands and output length `out_window`. If
		None, does not return controls.
	
	p: torch.tensor or None, shape=(n,) 
		A vector of probabilities that sum to 1 containing the sampling probability
		of each sequence. If None, use a uniform distribution.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 0.

	reverse_complement: bool, optional
		Whether to reverse complement-augment half of the data. Default is False.

	random_state: int or None, optional
		Whether to use a deterministic seed or not.
	"""

	def __init__(self, sequences, signals, controls=None, p=None, in_window=2114, 
		out_window=1000, max_jitter=0, reverse_complement=False, 
		random_state=None):
		self.p = p
		self.in_window = in_window
		self.out_window = out_window
		self.max_jitter = max_jitter
		
		self.reverse_complement = reverse_complement
		self.random_state = numpy.random.RandomState(random_state)

		self.signals = signals
		self.controls = controls
		self.sequences = sequences
		
		self.random_idxs = None
		self.n_random = 1000000

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self, idx):
		if idx % self.n_random == 0:
			self.random_idxs = self.random_state.choice(len(self), p=self.p, 
				size=self.n_random)

		i = self.random_idxs[idx % self.n_random]
		j = 0 if self.max_jitter == 0 else self.random_state.randint(
			self.max_jitter*2) 

		X = self.sequences[i][:, j:j+self.in_window]
		y = self.signals[i][:, j:j+self.out_window]

		if self.controls is not None:
			X_ctl = self.controls[i][:, j:j+self.in_window]

		if self.reverse_complement and self.random_state.choice(2) == 1:
			X = torch.flip(X, [0, 1])
			y = torch.flip(y, [0, 1])

			if self.controls is not None:
				X_ctl = torch.flip(X_ctl, [0, 1])

		if self.controls is not None:
			return X, X_ctl, y

		return X, y


def PeakGenerator(peaks, negatives, sequences, signals, controls=None,
	chroms=None, in_window=2114, out_window=1000, max_jitter=128, 
	negative_ratio=0.1, reverse_complement=True, shuffle=True, min_counts=None, 
	max_counts=None, summits=False, exclusion_lists=None, random_state=None, 
	pin_memory=True, num_workers=0, batch_size=32, verbose=False):
	"""This is a constructor function that handles all IO.

	This function will extract signal from all signal and control files,
	pass that into a DataGenerator, and wrap that using a PyTorch data
	loader. This is the only function that needs to be used.


	Parameters
	----------
	peaks: str or pandas.DataFrame or list/tuple of such
		A BED-formatted file containing peak coordinates. This can be either
		the string path to the BED file or a pandas DataFrame object containing
		three columns: chrom, start, and end. Alternatively, this can be a list
		of such objects whose coordinates will be interleaved.

	negatives: str or pandas.DataFrame or list/tuple of such
		A BED-formatted file containing negative coordinates. This can be either
		the string path to the BED file or a pandas DataFrame object containing
		three columns: chrom, start, and end. Alternatively, this can be a list
		of such objects whose coordinates will be interleaved.

	sequences: str or dictionary
		Either the path to a fasta file to read from or a dictionary where the
		keys are the unique set of chromosoms and the values are one-hot
		encoded sequences as numpy arrays or memory maps.

	signals: list of strs or list of dictionaries
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps.

	controls: list of strs or list of dictionaries or None, optional
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps. If None, no control tensor is returned. Default is None. 

	chroms: list or None, optional
		A set of chromosomes to extact loci from. Loci in other chromosomes
		in the locus file are ignored. If None, all loci are used. Default is
		None.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 128.

	negative_ratio: float, optional
		The ratio of negatives compared to peaks in each batch. A value of 1 means
		that each batch is balanced, and a value of 10 means that there would be 10
		negatives for each positive. Note that this is independent of the number of
		peaks and negatives provided. Even if the `peaks` input has 10x the number
		of coordinates as the `negatives` one, if the ratio is 1 each batch during
		training will be balanced (on average).

	reverse_complement: bool, optional
		Whether to reverse complement-augment half of the data. Default is True.

	shuffle: bool, optional
		Whether to randomly sample peaks, if True, or to proceed sequentially
		through them, if False. Negatives are always randomly sampled. Default
		is True.

	min_counts: float or None, optional
		The minimum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no minimum. Default 
		is None.

	max_counts: float or None, optional
		The maximum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no maximum. Default 
		is None.  

	summits: bool, optional
		Whether to return a region centered around the summit instead of the center
		between the start and end. If True, it will add the 10th column (index 9)
		to the start to get the center of the window, and so the data must be in 
		narrowPeak format.

	exclusion_lists: list or None, optional
		A list of strings of filenames to BED-formatted files containing exclusion
		lists, i.e., regions where overlapping loci should be filtered out. If None,
		no filtering is performed based on exclusion zones. Default is None.

	random_state: int or None, optional
		Whether to use a deterministic seed or not.

	pin_memory: bool, optional
		Whether to pin page memory to make data loading onto a GPU easier.
		Default is True.

	num_workers: int, optional
		The number of processes fetching data at a time to feed into a model.
		If 0, data is fetched from the main process. Default is 0.

	batch_size: int, optional
		The number of data elements per batch. Default is 32.
	
	verbose: bool, optional
		Whether to display a progress bar while loading. Default is False.


	Returns
	-------
	X: torch.utils.data.DataLoader
		A PyTorch DataLoader wrapped DataGenerator object.
	"""

	X_peaks = extract_loci(loci=peaks, sequences=sequences, 
		signals=signals, in_signals=controls, chroms=chroms, in_window=in_window, 
		out_window=out_window, max_jitter=max_jitter, min_counts=min_counts,
		max_counts=max_counts, summits=summits, exclusion_lists=exclusion_lists,
		ignore=list('QWERYUIOPSDFHJKLZXVBNM'), return_mask=True, verbose=verbose)

	loci_counts = X_peaks[1].sum(dim=(1, 2))
	
	outlier_threshold = torch.quantile(X_peaks[1].sum(dim=(1, 2)), 0.99) * 1.2
	outlier_idxs = loci_counts > outlier_threshold

	X_bg = extract_loci(loci=negatives, sequences=sequences, 
		signals=signals, in_signals=controls, chroms=chroms, in_window=in_window, 
		out_window=out_window, max_jitter=0, min_counts=min_counts,
		max_counts=max_counts, summits=False, exclusion_lists=exclusion_lists,
		ignore=list('QWERYUIOPSDFHJKLZXVBNM'), return_mask=True, verbose=verbose)

	if verbose:
		n_filtered_peaks = len(X_peaks[-1]) - X_peaks[-1].sum() + outlier_idxs.sum()
		n_filtered_negatives = len(X_bg[-1]) - X_bg[-1].sum()
		
		print("\nFiltered Peaks: {}".format(n_filtered_peaks))
		print("Filtered Negatives: {}".format(n_filtered_negatives))
	
	###

	X_gen = PeakNegativeSampler(
		peak_sequences=X_peaks[0][~outlier_idxs],
		peak_signals=X_peaks[1][~outlier_idxs],
		peak_controls=None if controls is None else X_peaks[2][~outlier_idxs],
		negative_sequences=X_bg[0],
		negative_signals=X_bg[1],
		negative_controls=None if controls is None else X_bg[2],
		negative_ratio=negative_ratio,
		in_window=in_window,
		out_window=out_window,
		max_jitter=max_jitter,
		reverse_complement=reverse_complement,
		shuffle=shuffle,
		random_state=random_state
	)

	X_gen = torch.utils.data.DataLoader(X_gen, pin_memory=pin_memory,
		num_workers=num_workers, batch_size=batch_size) 

	return X_gen
