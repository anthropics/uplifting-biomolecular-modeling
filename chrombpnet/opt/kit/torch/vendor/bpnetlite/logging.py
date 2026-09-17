# logging.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import pandas

class Logger():
	"""A logging class that can report or save metrics.

	This class contains a simple utility for saving statistics as they are
	generated, saving a report to a text file at the end, and optionally
	print the report to screen one line at a time as it is being generated.
	Must begin using the `start` method, which will reset the logger.
	
	
	Parameters
	----------
	names: list or tuple
		An iterable containing the names of the columns to be logged.
	
	verbose: bool, optional
		Whether to print to screen during the logging.
	"""

	def __init__(self, names, verbose=False):
		self.names = names
		self.verbose = verbose

	def start(self):
		"""Begin the recording process."""

		self.data = {name: [] for name in self.names}

		if self.verbose:
			print("\t".join(self.names))

	def add(self, row):
		"""Add a row to the log.

		This method will add one row to the log and, if verbosity is set,
		will print out the row to the log. The row must be the same length
		as the names given at instantiation.
		

		Parameters
		----------
		args: tuple or list
			An iterable containing the statistics to be saved.
		"""

		assert len(row) == len(self.names)

		for name, value in zip(self.names, row):
			self.data[name].append(value)

		if self.verbose:
			print("\t".join(map(str, [round(x, 4) if isinstance(x, float) else x 
				for x in row])))

	def save(self, name):
		"""Write a log to disk.

		
		Parameters
		----------
		name: str
			The filename to save the logs to.
		"""

		pandas.DataFrame(self.data).to_csv(name, sep='\t', index=False)
		