# chrombpnet.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import h5py
import time
import numpy
import torch

from .bpnet import BPNet
from .losses import MNLLLoss
from .losses import log1pMSELoss
from .performance import calculate_performance_measures
from .logging import Logger

from tqdm import trange

from tangermeme.predict import predict


class _Exp(torch.nn.Module):
	def __init__(self):
		super(_Exp, self).__init__()

	def forward(self, X):
		return torch.exp(X)


class _Log(torch.nn.Module):
	def __init__(self):
		super(_Log, self).__init__()

	def forward(self, X):
		return torch.log(X)


class ChromBPNet(torch.nn.Module):
	"""A ChromBPNet model.

	ChromBPNet is an extension of BPNet to handle chromatin accessibility data,
	in contrast to the protein binding data that BPNet handles. The distinction
	between these data types is that an enzyme used in DNase-seq and ATAC-seq
	experiments itself has a soft sequence preference, meaning that the
	strength of the signal is driven by real biology but that the exact read
	mapping locations are driven by the soft sequence bias of the enzyme.

	ChromBPNet handles this by treating the data using two models: a bias
	model that is initially trained on background (non-peak) regions where
	the bias dominates, and an accessibility model that is subsequently trained
	using a frozen version of the bias model. The bias model learns to remove
	the enzyme bias so that the accessibility model can learn real motifs.


	Parameters
	----------
	bias: torch.nn.Module 
		This model takes in sequence and outputs the shape one would expect in 
		ATAC-seq data due to Tn5 bias alone. This is usually a BPNet model
		from the bpnet-lite repo that has been trained on GC-matched non-peak
		regions.

	accessibility: torch.nn.Module
		This model takes in sequence and outputs the accessibility one would 
		expect due to the components of the sequence, but also takes in a cell 
		representation which modifies the parameters of the model, hence, 
		"dynamic." This model is usually a DynamicBPNet model, defined below.

	name: str, optional
		The name to prepend when saving the file. Default is ChromBPNet.
	"""

	def __init__(self, bias, accessibility, name="ChromBPNet"):
		super(ChromBPNet, self).__init__()
		for parameter in bias.parameters():
			parameter.requires_grad = False

		self.bias = bias
		self.accessibility = accessibility
		self.name = name
		self.logger = None
		self.n_control_tracks = accessibility.n_control_tracks
		self.n_outputs = 1
		self._log = _Log()
		self._exp1 = _Exp()
		self._exp2 = _Exp()


	def forward(self, X, X_ctl=None):
		"""A forward pass through the network.

		This function is usually accessed through calling the model, e.g.
		doing `model(x)`. The method defines how inputs are transformed into
		the outputs through interactions with each of the layers.


		Parameters
		----------
		X: torch.tensor, shape=(-1, 4, 2114)
			A one-hot encoded sequence tensor.

		X_ctl: ignore
			An ignored parameter for consistency with attribution functions.


		Returns
		-------
		y_profile: torch.tensor, shape=(-1, 1000)
			The predicted logit profile for each example. Note that this is not
			a normalized value.
		"""

		acc_profile, acc_counts = self.accessibility(X)
		bias_profile, bias_counts = self.bias(X)

		n0, n1 = acc_profile.shape[-1], bias_profile.shape[-1]
		w = (n1 - n0) // 2
		
		y_profile = acc_profile + bias_profile[:, :, w:-w]
		y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))
		
		return y_profile, y_counts


	def fit(self, training_data, optimizer, X_valid=None, y_valid=None,
		max_epochs=100, batch_size=64, validation_iter=100, dtype='float32',
		device='cuda', early_stopping=None, verbose=True):
		"""Fit the ChromBPNet model to data.

		Specifically, this function will fit the accessibility model to
		observed chromatin accessibility data, and assume that the bias model
		is frozen and pre-trained. Hence, the only parameters being trained
		in this function are those in the accessibility model.

		This function will save the best full ChromBPNet model, as well as the
		best accessibility model, found during training.


		Parameter
		---------
		training_data: torch.utils.data.DataLoader
			A data set that generates one-hot encoded sequence as input and
			read count signal for the output.

		optimizer: torch.optim.Optimizer
			A PyTorch optimizer.

		X_valid: torch.Tensor or None, shape=(-1, 4, length)
			A tensor of one-hot encoded sequences to use as input for the
			validation steps. If None, do not do validation. Default is None.

		y_valid: torch.Tensor or None, shape=(-1, 1, length)
			A tensor of read counts matched with the `X_valid` input. If None,
			do not do validation. Default is None.

		max_epochs: int
			The maximum number of training epochs to perform before stopping.
			Default is 100.

		batch_size: int
			The number of examples to use in each batch. Default is 64.

		validation_iter: int
			The number of training batches to perform before doing another
			round of validation. Set higher to spend a higher percentage of
			time in the training step.

		dtype: str or torch.dtype
			The torch.dtype to use when training. Usually, this will be torch.float32
			or torch.bfloat16. Default is torch.float32.
		
		device: str
			The device to use for training and inference. Typically, this will be
			'cuda' but can be anything supported by torch. Default is 'cuda'.

		early_stopping: int or None
			Whether to stop training early. If None, continue training until
			max_epochs is reached. If an integer, continue training until that
			number of `validation_iter` ticks has been hit without improvement
			in performance. Default is None.

		verbose: bool
			Whether to print the log as it is being generated. A log will
			be returned at the end of training regardless of this option, but
			when False, nothing will be printed to the screen during training.
			Default is False
		"""

		print("Warning: BPNet and ChromBPNet models trained using bpnet-lite may underperform those trained using the official repositories. See the GitHub README for further documentation.")
		
		dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
		
		y_bias_profile, y_bias_counts = predict(self.bias, X_valid, 
			batch_size=batch_size, dtype=dtype, device='cuda')

		self.logger = Logger(["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Training Count MSE", 
			"Validation MNLL", "Validation Profile Correlation", 
			"Validation Count Pearson", "Validation Count MSE", "Saved?"], 
			verbose=verbose)

		early_stop_count = 0
		start, best_loss = time.time(), float("inf")
		
		self.logger.start()
		self.bias.eval()
		for epoch in range(max_epochs):
			for iteration, (X, y) in enumerate(training_data):
				self.accessibility.train()

				X = X.cuda().float()
				y = y.cuda()

				optimizer.zero_grad()

				with torch.autocast(device_type=device, dtype=dtype):
					acc_profile, acc_counts = self.accessibility(X)
					bias_profile, bias_counts = self.bias(X)

					y_profile = torch.nn.functional.log_softmax(acc_profile +
						bias_profile, dim=-1)

					y_counts = torch.logsumexp(torch.stack([acc_counts, 
						bias_counts]), dim=0)

					profile_loss = MNLLLoss(y_profile, y).mean()
					count_loss = log1pMSELoss(y_counts, y.sum(dim=-1).reshape(-1, 
						1)).mean()

					profile_loss_ = profile_loss.item()
					count_loss_ = count_loss.item()

					loss = profile_loss + self.accessibility.alpha * count_loss
					loss.backward()
					optimizer.step()

				if verbose and iteration % validation_iter == 0:
					train_time = time.time() - start
					tic = time.time()

					with torch.no_grad():
						self.accessibility.eval()

						y_profile, y_counts = predict(self.accessibility, 
							X_valid, batch_size=batch_size, device=device,
							dtype=dtype)

						y_profile = torch.nn.functional.log_softmax(
							y_profile + y_bias_profile, dim=-1)

						y_counts = torch.logsumexp(torch.stack([y_counts,
							y_bias_counts]), dim=0)

						measures = calculate_performance_measures(y_profile, 
							y_valid, y_counts, kernel_sigma=7, 
							kernel_width=81, measures=['profile_mnll', 
							'profile_pearson', 'count_pearson', 'count_mse'])

						profile_corr = measures['profile_pearson']
						count_corr = measures['count_pearson']

						valid_loss = measures['profile_mnll'].mean()
						valid_loss += self.accessibility.alpha * measures['count_mse'].mean()
						valid_time = time.time() - tic

						self.logger.add([epoch, iteration, train_time, 
							valid_time, profile_loss_, count_loss_, 
							measures['profile_mnll'].mean().item(), 
							numpy.nan_to_num(profile_corr).mean(),
							numpy.nan_to_num(count_corr).mean(), 
							measures['count_mse'].mean().item(),
							(valid_loss < best_loss).item()])

						if valid_loss < best_loss:
							torch.save(self, "{}.torch".format(self.name))
							torch.save(self.accessibility, 
								"{}.accessibility.torch".format(self.name))
							best_loss = valid_loss
							early_stop_count = 0
						else:
							early_stop_count += 1

					start = time.time()

				if early_stopping is not None and early_stop_count >= early_stopping:
					break

			self.logger.save("{}.log".format(self.name))

			if early_stopping is not None and early_stop_count >= early_stopping:
				break

		torch.save(self, "{}.final.torch".format(self.name))
		torch.save(self, "{}.accessibility.final.torch".format(self.name))


	@classmethod
	def from_chrombpnet_lite(self, bias_model, accessibility_model, 
							 name='ChromBPNet'):
		"""Load a ChromBPNet model trained in ChromBPNet-lite.

		Confusingly, ChromBPNet-lite is a package written by Surag Nair that
		reorganized the ChromBPNet library and then was reintegrated back
		into it. However, some ChromBPNet models are still around that were
		trained using this package and this is a method for loading those
		models, not the models trained using the ChromBPNet package and not
		ChromBPNet models trained using this package.

		This method takes in paths to a h5 file containing the weights of the
		bias model and the accessibility model, both trained and whose outputs
		are organized according to TensorFlow. The weights are loaded and
		shaped into a PyTorch model and can be used as such.


		Parameters
		----------
		bias model: str
			The filename of the bias model.

		accessibility_model: str
			The filename of the accessibility model.

		name: str, optional
			The name to use when training the model and outputting to a file. Default
			is 'ChromBPNet'.
		

		Returns
		-------
		model: bpnetlite.models.ChromBPNet
			A PyTorch ChromBPNet model compatible with the bpnet-lite package.
		"""

		bias = BPNet.from_chrombpnet_lite(bias_model)
		acc = BPNet.from_chrombpnet_lite(accessibility_model)
		return ChromBPNet(bias, acc, name)


	@classmethod
	def from_chrombpnet(self, bias_model, accessibility_model, 
		name='ChromBPNet'):
		"""Load a ChromBPNet model trained using the official repository.

		This method takes in the path to a .h5 file containing the full model,
		i.e., the bias model AND the accessibility model. If you have two
		files -- one for the bias model, and one for the accessibility model --
		load them up as separate BPNet models and create a ChromBPNet object
		afterwards.


		Parameters
		----------
		bias model: str
			The filename of the bias model.

		accessibility_model: str
			The filename of the accessibility model.

		name: str, optional
			The name to use when training the model and outputting to a file.
		

		Returns
		-------
		model: bpnetlite.models.ChromBPNet
			A PyTorch ChromBPNet model compatible with the bpnet-lite package.
		"""


		bias = BPNet.from_chrombpnet(bias_model)
		acc = BPNet.from_chrombpnet(accessibility_model)
		return ChromBPNet(bias, acc, name)