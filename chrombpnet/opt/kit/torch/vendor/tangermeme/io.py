# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Code adapted from Alex Tseng, Avanti Shrikumar, and Ziga Avsec

from __future__ import annotations

import warnings

import numpy
import torch
import pandas

import pyfaidx
import pybigtools

from tqdm import tqdm

from .utils import one_hot_encode
from .utils import characters
from .utils import TangermemeWarning

from memelite.io import read_meme as memelite_read_meme


def _load_exclusion_zones(chrom_lengths, exclusion_lists):
	if exclusion_lists is not None:
		# Initialize the exclusion zones, where overlapping loci must be removed
		exclusion_zones = {}
		for chrom, size in chrom_lengths.items():
			exclusion_zones[chrom] = numpy.zeros(size // 100 + 1, dtype='bool')
		
		# Fill in the exclusion zones using the provided coordinates
		names = 'chrom', 'start', 'end'

		exclusion_list = pandas.concat([
			pandas.read_csv(elist, sep="\t", names=names, header=None, 
				usecols=(0, 1, 2)) for elist in exclusion_lists 
		])

		for _, (chrom, start, end) in exclusion_list.iterrows():
			start = start // 100
			end = end // 100 + 1

			exclusion_zones[chrom][start:end] = True
		
		return exclusion_zones


def _interleave_loci(loci, chroms=None, summits=False):
	"""An internal function for loading and processing the provided loci.

	There are two aspects that have to be considered when processing the loci.
	The first is that the user can pass in either strings containing filenames
	or pandas DataFrames. The second is that the user can pass in a single
	value or a list of values, and if a list of values the resulting dataframes
	must be interleaved.

	If a set of chromosomes is provided, each dataframe will be filtered to
	loci on those chromosomes before interleaving. If a more complicated form
	of filtering is desired, one should pre-filter the dataframes and pass those
	into this function for interleaving.

	If one wishes to center on summits, the input data must be in BED10 format
	where the 10th column (index 9 when zero-indexing) is the relative offset
	of the summit. This will adjust the start and end of the coordinates to
	be centered on the summits.


	Parameters
	----------
	loci: str, pandas.DataFrame, or list of those
		A filename to load, a pandas DataFrame in bed-format, or a list of
		either.

	chroms: list or None, optional
		A set of chromosomes to restrict the loci to. This is done before
		interleaving to ensure balance across sets of loci. If None, do not
		do filtering. Default is None.

	summits: bool, optional
		Whether to include the summits, which are the 10th column in a BED10 file,
		when loading the dataframe.


	Returns
	-------
	interleaved_loci: pandas.DataFrame
		A single pandas DataFrame that interleaves rows from each of the
		provided examples.
	"""

	if chroms is not None:
		if not isinstance(chroms, (list, tuple)):
			raise ValueError("Provided chroms must be a list.")

	if isinstance(loci, (str, pandas.DataFrame)):
		loci = [loci]
	elif not isinstance(loci, (list, tuple)):
		raise ValueError("Provided loci must be a string or pandas " +
			"DataFrame, or a list/tuple of those.")

	###
		
	cols = [0, 1, 2] + ([9] if summits else [])
	names = ['chrom', 'start', 'end'] + (['summit'] if summits else [])

	loci_dfs = []
	for i, df in enumerate(loci):
		# Extract the relevant columns from the dataframes
		if isinstance(df, str):
			df = pandas.read_csv(df, sep='\t', usecols=cols, 
				header=None, index_col=False, names=names)
		elif isinstance(df, pandas.DataFrame):
			df = df.iloc[:, cols].copy()
		else:
			raise ValueError("Provided loci must be a string or pandas " +
				"DataFrame, or a list/tuple of those.")

		# If using summits, correct the coordinates to be centered on them
		if summits:
			if df.iloc[:, -1].min() < 0:
				raise ValueError("Summits cannot be negative values.")

			if ((df.iloc[:, -1] + df.iloc[:, 1]) > df.iloc[:, 2]).any():
				raise ValueError("Summit + start cannot be larger than end.")

			mid = df['start'] + df['summit']
			w = df['end'] - df['start']
			
			df['start'] = mid - w // 2
			df['end'] = mid + w // 2
			df = df.drop(columns=['summit'])
		
		# If filtering chromosomes, remove loci on unallowed chromosomes
		if chroms is not None:
			df = df[numpy.isin(df['chrom'], chroms)]

		df['idx'] = numpy.arange(len(df)) * len(loci) + i
		loci_dfs.append(df)

	loci = pandas.concat(loci_dfs)
	loci = loci.set_index("idx").sort_index().reset_index(drop=True)
	return loci


def _load_signals(signals):
	"""An internal function for loading signals.

	The passed in signals must be a list but can either be a list of strings,
	which are interpreted as strings for bigwig files that should be opened,
	or dictionaries where the keys are chromosome names and the values are
	numpy arrays of values across the chromosome, which are kept as is.


	Parameters
	----------
	signals: list of strings or dicts or None
		A list of strings for bigwig files or dictionaries of numpy arrays.


	Returns
	-------
	_signals: list of dicts
		A list of either pointers to opened bigwig files or dictionaries of
		numpy arrays.
	"""

	if signals is None:
		return None

	_signals = []
	for i, signal in enumerate(signals):
		if isinstance(signal, str):
			signal = pybigtools.open(signal)
		elif not isinstance(signal, dict):
			raise ValueError("Signals must either be a list of strings " +
				"or a list of dictionaries.")
		elif not isinstance(list(signal.values())[0], numpy.ndarray):
			raise ValueError("Values in dictionaries must be numpy.ndarrays.")

		_signals.append(signal)

	return _signals


def _extract_locus_signal(signals, chrom, start, end):
	"""An internal function for extracting signal from a single locus.

	This function takes in a set of signals and a single locus and extracts
	the signal from each one of the loci.


	Parameters
	----------
	signals: list of pybigtools' BBIRead objects or dictionaries
		A list of BBIRead objects (as returned by pybigtools.open()) or dictionaries where the keys are
		chromosomes and the values are the signal at each position in the
		chromosome.

	chrom: str
		The name of the chromosome. Must be a key in the signals.

	start: int
		The starting coordinate to extract from, inclusive and base-0.

	end: int
		The ending coordinate to extract from, exclusive and base-0.


	Returns
	-------
	values: list of numpy.ndarrays, shape=(len(signals), end-start)
		The extracted signal from each of the signal files.

	Notes
	-----
	When `signal` is a dict, each `signal[chrom]` is assumed to be a 1-D
	array indexable by genomic position. Passing a `(n_tracks, length)`
	array under a single chromosome key will silently slice the first axis
	instead of positions and yield mis-shaped output; provide one signal
	dict per track in the outer `signals` list instead.
	"""

	if not isinstance(signals, (list, tuple)):
		raise ValueError("Provided signals must be in the form of a list.")

	values = []
	for i, signal in enumerate(signals):
		if isinstance(signal, dict):
			values_ = numpy.array(signal[chrom][start:end], dtype=numpy.float32)
		else:
			try:
				values_ = numpy.array(signal.values(chrom, start, end), dtype=numpy.float32)
			except (RuntimeError, ValueError, KeyError):
				warnings.warn(
					f"{chrom} {start} {end} not valid bigwig indexes. "
					"Using zeros instead.", TangermemeWarning, stacklevel=2)
				values_ = numpy.zeros(end-start, dtype=numpy.float32)
				
		values_ = numpy.nan_to_num(values_)
		values.append(values_)

	return values


def extract_loci(
	loci: str | list[str] | pandas.DataFrame | list[pandas.DataFrame],
	sequences: str | pyfaidx.Fasta | dict,
	signals: list | None = None,
	in_signals: list | None = None,
	chroms: list[str] | None = None,
	in_window: int = 2114,
	out_window: int = 1000,
	max_jitter: int = 0,
	min_counts: float | None = None,
	max_counts: float | None = None,
	target_idx: int = 0,
	n_loci: int | None = None,
	summits: bool = False,
	alphabet: list[str] = ['A', 'C', 'G', 'T'],
	ignore: list[str] = ['N'],
	exclusion_lists: list | None = None,
	return_mask: bool = False,
	verbose: bool = False,
) -> tuple:
	"""Extract sequence and signal information for each provided locus.

	This function will take in a set of loci, sequences, and optionally signals,
	and return the sequences and signals at each of the loci. Each of these
	parameters can be a filename, which is loaded internally, or an appropriate
	Python object (see below for details). The nomenclature `in/out` refers to
	the expected inputs and outputs of the downstream machine learning model,
	not this function.

	For each locus a sequence window of size `in_window` will be extracted from
	the sequences file and each of the `in_signals` files if provided, and
	a window of size `out_window` will be extracted from each of the `signals`
	files if provided. These windows are centered at the middle of the provided
	regions but will all be of the same size, regardless of the size of the peak.

	If `max_jitter` is provided, it will expand the windows for both the input
	and output. The results are not actually jittered, but this expanded window
	allows for downstream data generators to create jittered data while
	reducing the memory footprint of the returned data.

	There are a few reasons that the returned elements may not match one-to-one
	with the provided loci:

		- (1) If any of the coordinates fall off the end of chromosomes after
		accounting for jitter, the locus will be removed.

		- (2) If any of the loci fall on chromosomes not in a provided list,
		they will be removed.

		- (3) If min_counts or max_counts are specified and the locus has a
		number of counts not in those boundaries, the locus will be removed.

	If exclusion lists are provided, they will be used to filter out loci that
	fall in 100bp chunks that also include any of the regions in any of the
	exclusion lists. For example, if one of the exclusion lists has an element
	that is

		chr7    108    234

	loci will be removed if any of their bp fall within chr7 100 300.
 

	Parameters
	----------
	loci: str or pandas.DataFrame or list/tuple of such
		Either the path to a bed file or a pandas DataFrame object containing
		three columns: the chromosome, the start, and the end, of each locus
		to train on. Alternatively, a list or tuple of strings/DataFrames where
		the intention is to train on the interleaved concatenation, i.e., when
		you want to train on peaks and negatives.

	sequences: str or dictionary
		Either the path to a fasta file to read from or a dictionary where the
		keys are the unique set of chromosomes and the values are one-hot
		encoded sequences as numpy arrays or memory maps.

	signals: list of strs or list of dictionaries or None, optional
		A list of filepaths to bigwig files, where each filepath will be read
		using pybigtools, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps. If None, no signal tensor is returned. Default is None.

	in_signals: list of strs or list of dictionaries or None, optional
		A list of filepaths to bigwig files, where each filepath will be read
		using pybigtools, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps. If None, no tensor is returned. Default is None.

	chroms: list or None, optional
		A set of chromosomes to extract loci from. Loci in other chromosomes
		in the locus file are ignored. If None, all loci are used. Default is
		None.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 0.

	min_counts: float or None, optional
		The minimum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no minimum. Default 
		is None.

	max_counts: float or None, optional
		The maximum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no maximum. Default 
		is None.  

	target_idx: int, optional
		When specifying `min_counts` or `max_counts`, the single `signal`
		file to use when determining if a region has a number of counts in
		that range. Default is 0.

	n_loci: int or None, optional
		A cap on the number of loci to return. Note that this is not the
		number of loci that are considered. The difference is that some
		loci may be filtered out for various reasons, and those are not
		counted towards the total. If None, no cap. Default is None.

	summits: bool, optional
		Whether to return a region centered around the summit instead of the center
		between the start and end. If True, it will add the 10th column (index 9)
		to the start to get the center of the window, and so the data must be in 
		narrowPeak format.

	alphabet : set or tuple or list
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. Default is ['A', 'C', 'G', 'T'].

	ignore: list, optional
		A list of characters to ignore in the sequence, meaning that no bits
		are set to 1 in the returned one-hot encoding. Put another way, the
		sum across characters is equal to 1 for all positions except those
		where the original sequence is in this list. Default is ['N'].

	exclusion_lists: list or None, optional
		A list of strings of filenames to BED-formatted files containing exclusion
		lists, i.e., regions where overlapping loci should be filtered out. If None,
		no filtering is performed based on exclusion zones. Default is None.

	return_mask: bool, optional
		Whether to return a tensor containing whether each element in the provided
		loci have been filtered out because of falling off the edge of chromosomes
		or the signal not falling in the specified boundaries. Default is False.

	verbose: bool, optional
		Whether to display a progress bar while loading. Default is False.


	Returns
	-------
	seqs: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
		The extracted sequences in the same order as the loci in the locus
		file after optional filtering by chromosome.

	signals: torch.tensor, shape=(n, len(signals), out_window+2*max_jitter)
		The extracted signals where the first dimension is in the same order
		as loci in the locus file after optional filtering by chromosome and
		the second dimension is in the same order as the list of signal files.
		If no signal files are given, this is not returned.

	in_signals: torch.tensor, shape=(n, len(in_signals), in_window+2*max_jitter)
		The extracted in signals where the first dimension is in the same order
		as loci in the locus file after optional filtering by chromosome and
		the second dimension is in the same order as the list of in signal files.
		If no in signal files are given, this is not returned.

	kept_mask: torch.tensor, shape=(n0,), dtype=bool
		A boolean vector of length equal to the number of pre-filtered peaks, with
		entries being True if they were kept and False if they were filtered out.
		Applying this mask to the complete set of interleaved peaks will yield
		the returned values. Only returned if `return_mask=True`.
	"""

	seqs, signals_, in_signals_ = [], [], []
	kept_mask = []
	in_width, out_width = in_window // 2, out_window // 2
	if signals is None and in_signals is None:
		out_width = 0

	# Extract the length of each chromosome. Track whether we opened the
	# fasta ourselves so we know whether we are allowed to close it on
	# exit; a caller-provided pyfaidx.Fasta is theirs to manage.
	chrom_lengths = {}
	opened_fasta = False
	if isinstance(sequences, str):
		sequences = pyfaidx.Fasta(sequences)
		opened_fasta = True
		for key, value in sequences.items():
			chrom_lengths[str(key)] = len(value)
	elif isinstance(sequences, pyfaidx.Fasta):
		for key, value in sequences.items():
			chrom_lengths[str(key)] = len(value)
	else:
		for key, value in sequences.items():
			chrom_lengths[str(key)] = sequences[key].shape[-1]


	# Create the exclusion zones from the exclusion lists, if provided
	exclusion_zones = _load_exclusion_zones(chrom_lengths, exclusion_lists)

	# Load the loci
	loci = _interleave_loci(loci, chroms, summits=summits)
	
	signals = _load_signals(signals)
	in_signals = _load_signals(in_signals)
	
	desc = "Loading Loci"
	d = not verbose

	max_width = max(in_width, out_width)
	for chrom, start, end in tqdm(loci.values, disable=d, desc=desc):
		mid = start + (end - start) // 2

		start = mid - max(out_width, in_width) - max_jitter
		end = mid + max(out_width, in_width) + max_jitter

		# Does it fall off the end of a chromosome?
		if start < 0 or end > chrom_lengths[str(chrom)]:
			kept_mask.append(False)
			continue

		if exclusion_zones is not None:
			s, e = start // 100, end // 100 + 1
			if exclusion_zones[str(chrom)][s:e].any():
				kept_mask.append(False)
				continue

		# Extract a window of signal using the output size
		start = mid - out_width - max_jitter
		end = mid + out_width + max_jitter + (out_window % 2)

		if signals is not None:
			signal = _extract_locus_signal(signals, str(chrom), start, end)

			if min_counts is not None and signal[target_idx].sum() < min_counts:
				kept_mask.append(False)
				continue

			if max_counts is not None and signal[target_idx].sum() > max_counts:
				kept_mask.append(False)
				continue

			signals_.append(signal)

		# Extract a window of signal using the input size
		start = mid - in_width - max_jitter
		end = mid + in_width + max_jitter + (in_window % 2)

		if in_signals is not None:
			in_signal = _extract_locus_signal(in_signals, str(chrom), start, end)
			in_signals_.append(in_signal)

		# Extract a window of sequence using the input size
		if isinstance(sequences, dict):
			seq = sequences[str(chrom)][:, start:end]
		else:
			seq = one_hot_encode(sequences[str(chrom)][start:end].seq.upper(),
				alphabet=alphabet, ignore=ignore)

		kept_mask.append(True)
		seqs.append(seq)

		if n_loci is not None and len(seqs) == n_loci:
			break 

	if opened_fasta:
		sequences.close()
		
	# Figure out how to format the outputs depending on the provided parameters
	seqs = torch.from_numpy(numpy.stack(seqs))
	y_return = [seqs]

	if signals is not None:
		y_return.append(torch.from_numpy(numpy.stack(signals_)))

	if in_signals is not None:
		y_return.append(torch.from_numpy(numpy.stack(in_signals_)))
		
	if return_mask:
		kept_mask = torch.tensor(kept_mask)
		y_return.append(kept_mask)

	return y_return[0] if len(y_return) == 1 else y_return


def one_hot_to_fasta(
	X: torch.Tensor | numpy.ndarray,
	filename: str,
	mode: str = 'w',
	headers: list[str] | None = None,
	alphabet: list[str] = ['A', 'C', 'G', 'T'],
) -> None:
	"""Write out one-hot encoded sequences to a FASTA file.

	This function will take a set of one-hot encoded sequences and convert them
	to characters and write them out in FASTA format. If headers are provided
	for each sequence, these are used, otherwise the numeric index is used.


	Parameters
	----------
	X: torch.Tensor, shape=(-1, len(alphabet), length)
		A set of one-hot encoded sequences to write out.

	filename: str
		The path to the FASTA file to write to.

	mode: str, optional
		The file mode to open `filename` with, e.g. 'w' to overwrite or 'a' to
		append. Default is 'w'.

	headers: list of str or None, optional
		A list of one header per sequence in `X`. If None, the numeric index of
		each sequence is used as its header. Default is None.

	alphabet: set or tuple or list, optional
		A pre-defined alphabet where the ordering of the symbols is the same as
		the index into the one-hot encoding. Default is ['A', 'C', 'G', 'T'].
	"""
	
	with open(filename, mode=mode) as outfile:
		for i, X_seq in enumerate(X):
			X_chars = characters(X_seq, alphabet=alphabet)
			
			if headers is None:
				outfile.write("> {}\n".format(i))
			else:
				outfile.write("> {}\n".format(headers[i]))
			
			for start in range(0, len(X_chars), 80):
				outfile.write(X_chars[start:start+80] + "\n")
				
			outfile.write("\n")


def read_meme(filename: str, n_motifs: int | None = None) -> dict[str, torch.Tensor]:
	"""Read a MEME file and return a dictionary of PWMs.

	This method takes in the filename of a MEME-formatted file to read in
	and returns a dictionary of the PWMs where the keys are the metadata
	line and the values are the PWMs.

	This function is a wrapper around the memelite one, except that it returns
	torch tensors instead of numpy arrays.


	Parameters
	----------
	filename: str
		The filename of the MEME-formatted file to read in.

	n_motifs: int or None, optional
		If provided, stop reading after this many motifs have been parsed. If
		None, read all motifs in the file. Default is None.


	Returns
	-------
	motifs: dict
		A dictionary of the motifs in the MEME file.
	"""

	motifs = memelite_read_meme(filename, n_motifs=n_motifs)
	motifs = {name: torch.from_numpy(pwm) for name, pwm in motifs.items()}
	return motifs


def read_vcf(filename: str) -> pandas.DataFrame:
	"""Read a VCF file into a pandas DataFrame

	This function takes in the name of a file that is VCF formatted and returns
	a pandas DataFrame with the comments filtered out. This will only return the
	first 9 columns (CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT); any
	per-sample genotype columns past column 9 are silently dropped.

	Compressed VCFs are read transparently when pandas detects the compression
	from the filename extension (e.g., `.vcf.gz` works via `pandas.read_csv`).
	BCF (binary VCF) files are NOT supported by this function.


	Parameters
	----------
	filename: str
		The path to the VCF-formatted file to read in. May be plain `.vcf` or
		gzip-compressed `.vcf.gz`.


	Returns
	-------
	vcf: pandas.DataFrame
		A pandas DataFrame containing the rows, with columns CHROM, POS, ID,
		REF, ALT, QUAL, FILTER, INFO, FORMAT.
	"""

	names = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", 
		"FORMAT"]
	dtypes = {name: str for name in names}
	dtypes['POS'] = int

	vcf = pandas.read_csv(filename, delimiter='\t', comment='#', names=names, 
		dtype=dtypes, usecols=range(9))
	return vcf
