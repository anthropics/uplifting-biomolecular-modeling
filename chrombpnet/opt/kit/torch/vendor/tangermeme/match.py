# match.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Contribution by: Bo Vagner Hansen <bvh@bmb.sdu.dk>

"""
Provides functions for the calculation of GC-content genome-wide and the
sampling of GC-matched negatives.
"""

from __future__ import annotations

import numpy
import pandas
import pyfaidx
import pybigtools

from tqdm import tqdm
from scipy.stats import ks_2samp
from joblib import Parallel, delayed


def _chrom_coords_generator(chrom, chrom_size, width):
	"""Tiles a given `chrom` from 0 to `chrom_size`, returning region
	coords of size `width`. Produces chrom_size // width regions,
	ignoring the remainder."""
	start = 0
	for end in range(width, chrom_size+1, width):
		yield chrom, start, end
		start = end

def _resize_coords_generator(coords, width):
	"""Resizes the given `coords` to have size `width` and the same midpoint."""
	left_flank = width // 2
	right_flank = (width+1) // 2
	for chrom, start, end in coords:
		mid = start + (end-start) // 2
		yield chrom, mid-left_flank, mid+right_flank

def _loci_coords_generator(loci_df, width = None):
	"""Takes in a pandas dataframe `loci_df` and returns a generator of loci coordinates.
	The dataframe must contain columns named chrom, start and end - other columns are not
	used. If `width` is set to None, will use the start and end coordinates as given.
	if `width` is an integer, will return start and end coordinates that are `width` apart,
	and centered on the midpoint of the given start and end coordinates."""
	g = ((locus.chrom, locus.start, locus.end) for locus in loci_df.itertuples(index=False))
	return g if width is None else _resize_coords_generator(g, width)

def _valid_locus(chrom, start, end, chrom_sizes):
	"""Returns a bool telling whether the specified locus is valid, i.e., the `chrom`
	is found in the `chrom_sizes` dictionary and `start` and `end` are within bounds."""
	return (chrom in chrom_sizes) and (start >= 0) and (end <= chrom_sizes[chrom])

def _valid_generator(coords, chrom_sizes):
	"""Returns a generator filtering the provided `coords` for invalid loci using
	the `chrom_sizes` dictionary."""
	return (locus for locus in coords if _valid_locus(*locus, chrom_sizes))

def _sequence_generator_slice(coords, genome):
	"""Generates the sequences given by `coords` from the sliceable `genome`,
	e.g. a str, list or tuple."""
	return (genome[start:end] for _,start,end in coords)

def _sequence_generator_stream(coords, genome_stream):
	"""Streams the sequences given by `coords` from the pyfaidx `genome_stream`.
	Returns a generator, so that only one sequence is kept in memory at a time
	when passing the generator on to an iterator that consumes the sequence."""
	return (genome_stream[chrom][start:end].seq.upper() for chrom,start,end in coords)

def _sequence_generator_buffered(coords, genome_stream):
	"""Returns a generator of sequences given by `coords` from the pyfaidx `genome_stream`.
	The entire chromosomal sequence is loaded into memory. Should mainly be used when
	there are many coords on each chromosome. If not all coords are on the same chromosome,
	the coords should definitely be sorted, as otherwise the function will be very inefficient."""
	buffer_chrom = ''
	for chrom, start, end in coords:
		if buffer_chrom != chrom:
			buffer_chrom = chrom
			buffer_seq = genome_stream[chrom][:].seq
		yield buffer_seq[start:end].upper()

def _sequence_generator(coords, genome, buffer = False):
	"""Wrapper function returning the sequence generator corresponding to the value of
	`genome` and `buffer`. Takes in a `genome`, which can be either a pyfaidx.Fasta object
	or a sliceable object like a str, list or tuple. `buffer` is only used for pyfaidx.Fasta
	objects. `buffer` determines whether the sequences should be streamed (False) or extracted
	as slices from a buffered chromosomal sequence (True)."""
	if isinstance(genome, pyfaidx.Fasta):
		if buffer:
			return _sequence_generator_buffered(coords, genome)
		else:
			return _sequence_generator_stream(coords, genome)
	else:
		return _sequence_generator_slice(coords, genome)

def _perc_generator(sequences, chars):
	"""Calculates the percentage (as a decimal) of `chars` in the provided `sequences`.
	Returns a generator of these percentages."""
	return (sum(seq.count(c) for c in chars) / len(seq) for seq in sequences)

def _extract_counts(chrom, start, end, bw_stream):
	"""Extract the signal from the pybigtools `bw_stream` in the provided locus and returns
	the sum of the base pair values. If the signal cannot be extracted, returns nan."""
	try:
		value = bw_stream.values(chrom, start, end, bins=1, summary="sum", exact=True)[0]
		return value if value is not None else 0
	except (RuntimeError, ValueError, KeyError, IndexError):
		return float('nan')

def _count_generator(coords, bw_stream, buffer = False):
	"""Wrapper function returning the count generator corresponding to the value of `buffer`.
	If no signal can be extracted from a locus, nan is returned for that locus.
	Buffer determines whether the signals should be streamed (False) or extracted as slices
	from a buffered chromosomal signal (True)."""
	if buffer:
		return _count_generator_buffered(coords, bw_stream)
	else:
		return _count_generator_stream(coords, bw_stream)

def _count_generator_stream(coords, bw_stream):
	"""Takes a list of locus `coords` and a pybigtools `bw_stream` and returns a generator
	producing the sum of counts for each locus. The signals are streamed from the locus,
	such that only one locus signal is kept in memory at a time. As reading from a bigwig
	is a bit slow, it can be quite time consuming to use this function across an entire
	chromosome, hence the buffered version should be preferred in that case."""
	return (_extract_counts(*locus, bw_stream) for locus in coords)

def _count_generator_buffered(coords, bw_stream):
	"""Takes a list of locus `coords` and a pybigtools `bw_stream` and returns a generator
	producing the sum of counts for each locus. The signal for an entire chromosome
	is kept in memory at a time, to speed up the extraction of multiple signals across
	a chromosome. This function should mainly be used if calculating the counts for
	densely spaced regions across the chromosomes. In addition, `coords` should be sorted
	by chromosome, otherwise the function will be very inefficient."""
	buffer_chrom = ''
	for chrom, start, end in coords:
		if buffer_chrom != chrom:
			buffer_chrom = chrom
			try:
				buffer_signal = bw_stream.values(chrom, 0, None)
			except (RuntimeError, ValueError, KeyError):
				buffer_signal = None
		if buffer_signal is not None:
			yield numpy.nansum(buffer_signal[start:end]).item()
		else:
			yield float('nan')

def _char_perc_from_coords(fasta, coords, chars, num_regions=-1, buffer=False, verbose=False):
	"""This method will take in a `fasta` file and return the percentage of `chars`
	in the sequences extracted from the fasta file and defined by the list of `coords`.
	This is usually used to calculate GC percentage but can also be used to calculate
	the percentage of N's.

	Parameters
	----------
	fasta: str
		The path to a fasta file, usually a reference genome.

	coords: list, tuple or generator of such
		iterable of tuples formatted like (chr, start, end),
		where `chr` is a string and `start` and `end` are integers.

	chars: iterable, such as set, list, tuple or str
		The characters to look for in the sequences bounded by `coords`.
		The function returns the percentage of these characters found in
		each sequence.

	num_regions: int, optional
		The number of regions given by coords, i.e. the length of coords.
		Used for efficient construction of the output numpy array, in case
		coords is given as a generator. If set to -1, the number or regions
		will automatically be inferred. Default is -1.

	buffer: bool, optional
		Whether to load the entire chromosomal sequence into memory and
		and extract the sequences as slices of the entire chromosome.
		Should only be set to true if there are many regions
		on the same chromosome, such as when calculating GC content for
		an entire chromosome. It is not recommend to use buffering if
		calculating char percentages for only peak regions, but if done
		anyway for some reason, the peaks should be sorted by chromosome.
		If set to false, will instead only load the sequence for a single
		region at a time into memory. Default is False.

	verbose: bool, optional
		Whether to display a progress bar.

	Returns
	-------
	perc: numpy.ndarray, shape=(len(list(coords)), )
	"""
	desc = "Getting %s percentages" % ''.join(chars)

	with pyfaidx.Fasta(fasta) as genome_stream:
		generator = _sequence_generator(coords, genome_stream, buffer=buffer)
		generator = _perc_generator(generator, chars)
		generator = tqdm(generator, disable=not verbose, desc=desc)
		perc = numpy.fromiter(generator, dtype=float, count=num_regions)

	return perc

def _counts_from_coords(bigwig, coords, num_regions=-1, buffer=False, verbose=False):
	"""An internal function returning the summed signal in each region of `coords`
	extracted from the `bigwig` file.

	This method will take in a `bigwig` file and return the summed signal in
	each region defined by `coords`. If signal cannot be extracted from a
	region, NaN is returned for that region.

	Parameters
	----------
	bigwig: str
		The path to a bigwig file to extract counts from.

	coords: list, tuple or generator of such
		iterable of tuples formatted like (chr, start, end),
		where `chr` is a string and `start` and `end` are integers.

	num_regions: int, optional
		The number of regions given by coords, i.e. the length of coords.
		Used for efficient construction of the output numpy array, in case
		coords is given as a generator. If set to -1, the number or regions
		will automatically be inferred. Default is -1.

	buffer: bool, optional
		Whether to load the entire chromosomal signal into memory and
		and extract the locus signals as slices of the entire chromosome.
		Should only be set to true if there are many regions on the same
		chromosome. If calculating counts for regions on many chromosomes,
		the regions should be sorted by chromosome.
		If set to false, will instead only load the signal for a single
  		region at a time into memory. Default is False.

	verbose: bool, optional
		Whether to display a progress bar.

	Returns
	-------
	count: numpy.ndarray, shape=(len(list(coords)), )
	"""
	desc = "Getting counts"

	with pybigtools.open(bigwig, "r") as bw_stream:
		generator = _count_generator(coords, bw_stream, buffer=buffer)
		generator = tqdm(generator, disable=not verbose, desc=desc)
		count = numpy.fromiter(generator, dtype=float, count=num_regions)

	return count

def _calculate_char_perc(sequence, width, chars):
	"""An internal function returning the percentage of `chars` in `sequence`.

	This method will take in a string `sequence` and return the percentage
	of each non-overlapping block of length `width` across the sequence
	containing `chars`. This is usually used to calculate GC percentage but
	can also be used to calculate the percentage of N's in a sequence.


	Parameters
	----------
	sequence: str
		A string made up of the alphabet 'A', 'C', 'G', 'T', 'N'.

	width: int
		The total width of the window to calculate the GC content for, e.g.,
		`1000` if you want to calculate this for 1000 bp blocks.

	chars: iterable, such as set, list, tuple or str
		The characters to look for in the sequence. The returned percentage is
		the percentage of the original sequence that is one of these characters.

	Returns
	-------
	perc: numpy.ndarray, shape=(len(sequence) // width,)
	"""

	seq_len = len(sequence)
	num_regions = seq_len // width
	coords = _chrom_coords_generator(chrom='', chrom_size=seq_len, width=width)
	seqs = _sequence_generator(coords, sequence)
	perc = _perc_generator(seqs, chars)
	perc = numpy.fromiter(perc, dtype=float, count=num_regions)
	return perc

def _extract_and_filter_chrom(fasta, chrom, in_window, out_window,
	max_n_perc=0.1, gc_bin_width=0.02, bigwig=None, signal_threshold=None):
	"""Calculate GC content for, and filter, one chromosome.

	This function will take in the name of a FASTA file, a chromosome, and a
	percentage of Ns that cannot be exceeded, and return a set of passing loci
	and their exact GC percentage. Optionally, it will also take in a bigwig
	and filter the loci based on having a signal threshold smaller than some
	value, where the signal is summed across the width.


	Parameters
	----------
	fasta: str
		The filepath to the FASTA file to extract sequences from.

	chrom: str
		The chromosome to extract from the FASTA file. Must be in the file.

	in_window: int
		The window to calculate the GC content over, corresponding to the input
		window of the downstream model that will be trained.

	out_window: int
		The window to calculate signal for and apply the signal threshold to,
		corresponding to the output window of the downstream model that will
		be trained.

	max_n_perc: float, range=(0, 1.0), optional
		The maximum percentage of N characters in each window to be considered.
		All windows with a higher percentage are discarded. Default is 0.1.

	gc_bin_width: float, range=(0, 1.0), optional
		The bin size to discretize GC content. Default is 0.02.

	bigwig: str or None, optional
		If filtering regions based on signal strength, calculate the signal
		from this bigwig. If None, do not filter based on signal strength.
		Default is None.

	signal_threshold: float or None, optional
		The maximum possible signal, summed across the entire window, that
		each window can have without being filtered. If the window has a summed
		signal higher than this value, the window is discarded. Default is None.

	Returns
	-------
	gc_percs: dict
		A dictionary where the keys are observed GC bins and the values are
		lists of loci that are in that GC bin. Each returned locus is the index
		not the true value and so corresponds to the real position integer
		divided by in_window.
	"""

	with pyfaidx.Fasta(fasta) as f:
		sequence = f[chrom][:].seq.upper()

	gc_perc = _calculate_char_perc(sequence, in_window, 'GC')
	n_perc = _calculate_char_perc(sequence, in_window, 'N')
	del sequence

	idxs = n_perc <= max_n_perc

	if bigwig is not None:
		assert(in_window >= out_window), "out_window cannot be larger than in_window."
		left_flank = (in_window - out_window) // 2
		right_flank = (in_window - out_window + 1) // 2

		with pybigtools.open(bigwig, "r") as bw:
			try:
				values = bw.values(chrom, 0, None)
			except RuntimeError:
				return {}

		values = values[:values.shape[0] // in_window * in_window]
		values = values.reshape(-1, in_window)
		values = values[:, left_flank:-right_flank]
		values = numpy.nansum(values, axis=-1)

		idxs = idxs & (values <= signal_threshold)

	gc_perc = ((gc_perc + gc_bin_width / 2.) // gc_bin_width).astype(int)
	unique_gc = numpy.unique(gc_perc[idxs]).tolist()
	gc_perc = {gc:numpy.nonzero(idxs & (gc_perc == gc))[0].tolist() for gc in unique_gc}

	return gc_perc


def extract_matching_loci(
	loci: str | pandas.DataFrame,
	fasta: str,
	in_window: int = 2114,
	out_window: int = 1000,
	max_n_perc: float = 0.1,
	gc_bin_width: float = 0.02,
	bigwig: str | None = None,
	signal_beta: float = 0.5,
	chroms: list[str] | None = None,
	random_state: int | numpy.random.RandomState | None = None,
	n_jobs: int = 1,
	verbose: bool = False,
) -> pandas.DataFrame:
	"""Extract matching loci given a fasta file.

	This function takes in a set of loci (a bed file or a pandas dataframe in
	bed format) and returns a GC-matched set of negatives. This will also
	perform basic filtering to ignore regions of the genome that are too high
	in Ns. Optionally, it can take in a bigwig and a signal threshold and only
	select regions that have fewer than a threshold of counts in each region.

	Importantly, it will apply `max_n_perc` to both the loci that are passed in
	and also potential regions that can be selected. This means that if a locus
	passed in has higher than `max_n_perc` number of unspecified positions,
	it will be filtered out, and a smaller number of positions will be selected.
	This is done because the GC content of a region with many Ns in it is not
	trustworthy.


	Parameters
	----------
	loci: str or pandas dataframe
		A filepath to a bed file, or a pandas dataframe in bed format.

	fasta: str
		The filepath to the FASTA file to extract sequences from.

	in_window: int
		The window to calculate the GC content over, corresponding to the input
		window of the downstream model that will be trained.

	out_window: int
		The window to calculate signal for and apply the signal threshold to,
		corresponding to the output window of the downstream model that will
		be trained.

	max_n_perc: float, range=(0, 1.0), optional
		The maximum percentage of N characters in each window to be considered.
		All windows with a higher percentage are discarded. Default is 0.1.

	gc_bin_width: float, range=(0, 1.0), optional
		The bin size to discretize GC content. Default is 0.02.

	bigwig: str or None, optional
		If filtering regions based on signal strength, calculate the signal
		from this bigwig. If None, do not filter based on signal strength.
		Default is None.

	signal_beta: float, optional
		A multiplier of the robust minimum signal calculated from `loci` that
		each background region must have fewer reads than. Only relevant if a
		bigwig is passed in. Must be a number when `bigwig` is set (the code
		computes `robust_min * signal_beta`); passing `None` together with a
		`bigwig` raises a `TypeError`. Default is 0.5.

	chroms: list, tuple, or None, optional
		A set of chromosomes to use when choosing matching loci. If None, only
		use chromosomes that the loci themselves are drawn from. Default is
		None.

	random_state: numpy.random.RandomState, int or None, optional
		A random state to use for sampling loci. If a RandomState object or
		an integer, this will produce deterministic sampling. If None, sampling
		will be different each time. Default is None.

	n_jobs: integer, optional
		Number of parallel processes to use for extracting background GC content.
		-1 means use all available CPUs. Default is 1.

	verbose: bool, optional
		Whether to print display bars and diagnostics to ensure that the
		sampling is reasonable. When set to True, there may be a large amount
		of output. Default is False.


	Returns
	-------
	matched_loci: pandas.DataFrame
		A bed-formatted set of matched loci sorted first by chromosome and
		then by position on the chromosome. Note that these are not sorted
		such that the i-th position in this file is a GC match for the i-th
		position in the original locus file.
	"""

	if not isinstance(random_state, numpy.random.RandomState):
		random_state = numpy.random.RandomState(random_state)

	if isinstance(loci, str):
		loci = pandas.read_csv(loci, sep='\t', usecols=[0, 1, 2], header=None,
			index_col=False, names=['chrom', 'start', 'end'])

	if chroms is not None:
		loci = loci[numpy.isin(loci['chrom'], chroms)]
	else:
		chroms = numpy.unique(loci['chrom'])

	fa = pyfaidx.Fasta(fasta)
	chrom_sizes = {key: len(fa[key]) for key in chroms}

	if verbose: 
		print("Processing given loci.")
	
	coords = _loci_coords_generator(loci, max(in_window, out_window))
	coords = list(_valid_generator(coords, chrom_sizes))
	num_regions = len(coords)

	threshold = None
	if bigwig is not None:
		coords = list(_resize_coords_generator(coords, out_window))
		loci_count = _counts_from_coords(bigwig, coords, num_regions, buffer=False, verbose=verbose)
		robust_min = numpy.nanquantile(loci_count, 0.01).item()
		threshold = robust_min * signal_beta

	coords = list(_resize_coords_generator(coords, in_window))
	loci_n = _char_perc_from_coords(fasta, coords, 'N', num_regions, buffer=False, verbose=verbose)
	loci_gc = _char_perc_from_coords(fasta, coords, 'GC', num_regions, buffer=False, verbose=verbose)
	loci_gc = loci_gc[loci_n < max_n_perc]

	loci_gc = ((loci_gc + gc_bin_width / 2.) // gc_bin_width).astype(int)
	loci_bin_count = numpy.zeros(int(1./gc_bin_width)+1, dtype=int)
	for gc_bin in loci_gc:
		loci_bin_count[gc_bin] += 1

	# Extract mask of already-selected loci
	mask = {chrom: [] for chrom in chroms}
	for locus in loci.itertuples(index=False):
		if locus.chrom not in mask:
			continue

		start = locus.start // in_window
		end = locus.end // in_window + 1

		mask[locus.chrom].extend(range(start, end))

	for chrom, values in mask.items():
		mask[chrom] = set(values)

	# Get GC content of background regions
	desc = 'Getting background GC'
	f = delayed(_extract_and_filter_chrom)
	chrom_percs = Parallel(n_jobs=n_jobs)(f(
		fasta=fasta,
		chrom=chrom,
		in_window=in_window,
		out_window=out_window,
		max_n_perc=max_n_perc,
		gc_bin_width=gc_bin_width,
		bigwig=bigwig,
		signal_threshold=threshold)
		for chrom in tqdm(chroms, disable=not verbose, desc=desc))

	# Merge them into a single dictionary, keeping track of chroms
	bg_bin_count = numpy.zeros(int(1./gc_bin_width) + 1, dtype=int)
	gc_percs = {perc: [] for perc in range(len(bg_bin_count))}

	for chrom, percs in zip(chroms, chrom_percs):
		for key, values in percs.items():
			for value in values:
				if value not in mask[chrom]:
					gc_percs[key].append((chrom, value))
					bg_bin_count[key] += 1

	for key, value in gc_percs.items():
		random_state.shuffle(value)

	orig_bg_bin_count = bg_bin_count.copy()
	orig_loci_bin_count = loci_bin_count.copy()

	# Match the sizes
	matched_loci_bin_count = numpy.minimum(bg_bin_count, loci_bin_count)
	bg_bin_count -= matched_loci_bin_count
	loci_bin_count -= matched_loci_bin_count

	n = len(loci_bin_count)
	for i in range(n-1, -1, -1):
		if loci_bin_count[i] == 0:
			continue

		for offset in range(n):
			idx = i + offset
			if idx < n:
				count = min(bg_bin_count[idx], loci_bin_count[i])
				bg_bin_count[idx] -= count
				loci_bin_count[i] -= count
				matched_loci_bin_count[idx] += count

			if loci_bin_count[i] == 0:
				break

			idx = i - offset
			if idx >= 0:
				count = min(bg_bin_count[idx], loci_bin_count[i])
				bg_bin_count[idx] -= count
				loci_bin_count[i] -= count
				matched_loci_bin_count[idx] += count

			if loci_bin_count[i] == 0:
				break

	if verbose:
		numpy.set_printoptions(suppress=True)
		print("GC Bin\tBackground Count\tPeak Count\tChosen Count")
		for i in range(n):
			print("{:2.2f}: {:8d}\t{:8d}\t{:8d}".format(
				numpy.arange(0, 1.01, gc_bin_width)[i],
				orig_bg_bin_count[i], orig_loci_bin_count[i],
				matched_loci_bin_count[i]))

	# Extract the loci
	matched_loci = {'chrom': [], 'start': [], 'end': []}
	for i in range(n):
		for j in range(matched_loci_bin_count[i]):
			chrom, start = gc_percs[i][j]

			matched_loci['chrom'].append(chrom)
			matched_loci['start'].append(start*in_window)
			matched_loci['end'].append((start+1)*in_window)

	matched_loci = pandas.DataFrame(matched_loci)

	if verbose:
		matched_gc = []
		for i,j in enumerate(matched_loci_bin_count):
			matched_gc.extend([i]*j)

		stats = ks_2samp(loci_gc, matched_gc)
		print("GC-bin KS test stat:{:3.3}, p-value {:3.3}".format(
			stats.statistic, stats.pvalue))

		if bigwig is not None:
			print("Processing matched loci.")
			coords = _loci_coords_generator(matched_loci, out_window)
			num_regions = len(matched_loci)
			matched_count_max = _counts_from_coords(bigwig, coords, num_regions, buffer=False, verbose=verbose).max()
			print("Peak Robust Signal Minimum: {}".format(robust_min))
			print("Matched Signal Maximum: {}".format(matched_count_max))

	matched_loci = matched_loci.sort_values(["chrom", "start"])
	return matched_loci
