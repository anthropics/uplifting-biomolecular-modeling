# bpnet.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This module contains a reference implementation of BPNet that can be used
or adapted for your own circumstances. The implementation takes in a
stranded control track and makes predictions for stranded outputs.
"""

import h5py
import time 
import numpy
import torch

from .losses import MNLLLoss
from .losses import log1pMSELoss
from .losses import _mixture_loss

from .performance import pearson_corr
from .performance import calculate_performance_measures
from .logging import Logger

from tqdm import tqdm

from tangermeme.predict import predict

#torch.backends.cudnn.benchmark = True	


class ControlWrapper(torch.nn.Module):
	"""This wrapper automatically creates a control track of all zeroes.

	This wrapper will check to see whether the model is expecting a control
	track (e.g., most BPNet-style models) and will create one with the expected
	shape. If no control track is expected then it will provide the normal
	output from the model.
	"""

	def __init__(self, model):
		super(ControlWrapper, self).__init__()
		self.model = model

	def forward(self, X, X_ctl=None):
		if X_ctl != None:
			return self.model(X, X_ctl)

		if self.model.n_control_tracks == 0:
			return self.model(X)

		X_ctl = torch.zeros(X.shape[0], self.model.n_control_tracks,
			X.shape[-1], dtype=X.dtype, device=X.device)
		return self.model(X, X_ctl)
	

class _ProfileLogitScaling(torch.nn.Module):
	"""This ugly class is necessary because of Captum.

	Captum internally registers classes as linear or non-linear. Because the
	profile wrapper performs some non-linear operations, those operations must
	be registered as such. However, the inputs to the wrapper are not the
	logits that are being modified in a non-linear manner but rather the
	original sequence that is subsequently run through the model. Hence, this
	object will contain all of the operations performed on the logits and
	can be registered.


	Parameters
	----------
	logits: torch.Tensor, shape=(-1, -1)
		The logits as they come out of a Chrom/BPNet model.
	"""

	def __init__(self):
		super(_ProfileLogitScaling, self).__init__()
		self.softmax = torch.nn.Softmax(dim=-1)

	def forward(self, logits):
		y_softmax = self.softmax(logits)
		return logits * y_softmax


class ProfileWrapper(torch.nn.Module):
	"""A wrapper class that returns transformed profiles.

	This class takes in a trained model and returns the weighted softmaxed
	outputs of the first dimension. Specifically, it takes the predicted
	"logits" and takes the dot product between them and the softmaxed versions
	of those logits. This is for convenience when using captum to calculate
	attribution scores.

	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super(ProfileWrapper, self).__init__()
		self.model = model
		self.flatten = torch.nn.Flatten()
		self.scaling = _ProfileLogitScaling()

	def forward(self, X, X_ctl=None, **kwargs):
		logits = self.model(X, X_ctl, **kwargs)[0]
		logits = self.flatten(logits)
		logits = logits - torch.mean(logits, dim=-1, keepdims=True)
		return self.scaling(logits).sum(dim=-1, keepdims=True)


class CountWrapper(torch.nn.Module):
	"""A wrapper class that only returns the predicted counts.

	This class takes in a trained model and returns only the second output.
	For BPNet models, this means that it is only returning the count
	predictions. This is for convenience when using captum to calculate
	attribution scores.

	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super(CountWrapper, self).__init__()
		self.model = model

	def forward(self, X, X_ctl=None, **kwargs):
		return self.model(X, X_ctl, **kwargs)[1]


class BPNet(torch.nn.Module):
	"""A basic BPNet model with stranded profile and total count prediction.

	This is a reference implementation for BPNet models. It exactly matches the
	architecture in the official ChromBPNet repository. It is very similar to
	the implementation in the official basepairmodels repository but differs in
	when the activation function is applied for the resifual layers. See the
	BasePairNet object below for an implementation that matches that repository. 

	The model takes in one-hot encoded sequence, runs it through: 

	(1) a single wide convolution operation 

	THEN 

	(2) a user-defined number of dilated residual convolutions

	THEN

	(3a) profile predictions done using a very wide convolution layer 
	that also takes in stranded control tracks 

	AND

	(3b) total count prediction done using an average pooling on the output
	from 2 followed by concatenation with the log1p of the sum of the
	stranded control tracks and then run through a dense layer.

	This implementation differs from the original BPNet implementation in
	two ways:

	(1) The model concatenates stranded control tracks for profile
	prediction as opposed to adding the two strands together and also then
	smoothing that track 

	(2) The control input for the count prediction task is the log1p of
	the strand-wise sum of the control tracks, as opposed to the raw
	counts themselves.

	(3) A single log softmax is applied across both strands such that
	the logsumexp of both strands together is 0. Put another way, the
	two strands are concatenated together, a log softmax is applied,
	and the MNLL loss is calculated on the concatenation. 

	(4) The count prediction task is predicting the total counts across
	both strands. The counts are then distributed across strands according
	to the single log softmax from 3.


	Parameters
	----------
	n_filters: int, optional
		The number of filters to use per convolution. Default is 64.

	n_layers: int, optional
		The number of dilated residual layers to include in the model.
		Default is 8.

	n_outputs: int, optional
		The number of profile outputs from the model. Generally either 1 or 2 
		depending on if the data is unstranded or stranded. Default is 2.

	n_control_tracks: int, optional
		The number of control tracks to feed into the model. When predicting
		TFs, this is usually 2. When predicting accessibility, this is usualy
		0. When 0, this input is removed from the model. Default is 2.

	count_loss_weight: float, optional
		The weight to put on the count loss.

	profile_output_bias: bool, optional
		Whether to include a bias term in the final profile convolution.
		Removing this term can help with attribution stability and will usually
		not affect performance. Default is True.

	count_output_bias: bool, optional
		Whether to include a bias term in the linear layer used to predict
		counts. Removing this term can help with attribution stability but
		may affect performance. Default is True.

	name: str or None, optional
		The name to save the model to during training.

	trimming: int or None, optional
		The amount to trim from both sides of the input window to get the
		output window. This value is removed from both sides, so the total
		number of positions removed is 2*trimming.

	verbose: bool, optional
		Whether to display statistics during training. Setting this to False
		will still save the file at the end, but does not print anything to
		screen during training. Default is True.
	"""

	def __init__(self, n_filters=64, n_layers=8, n_outputs=2, 
		n_control_tracks=2, count_loss_weight=1, profile_output_bias=True,
		count_output_bias=True, name=None, trimming=None, verbose=True):
		super(BPNet, self).__init__()
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_outputs = n_outputs
		self.n_control_tracks = n_control_tracks

		self.count_loss_weight = count_loss_weight
		self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)
		self.trimming = trimming or 47 + sum(2**i for i in range(1, n_layers+1))

		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
		self.irelu = torch.nn.ReLU()

		self.rconvs = torch.nn.ModuleList([
			torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
				dilation=2**i) for i in range(1, self.n_layers+1)
		])
		self.rrelus = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(1, self.n_layers+1)
		])

		self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs, 
			kernel_size=75, padding=37, bias=profile_output_bias)
		
		n_count_control = 1 if n_control_tracks > 0 else 0
		self.linear = torch.nn.Linear(n_filters+n_count_control, 1, 
			bias=count_output_bias)

		self.logger = Logger(["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Training Count MSE", 
			"Validation MNLL", "Validation Profile Pearson", 
			"Validation Count Pearson", "Validation Count MSE", "Saved?"], 
			verbose=verbose)


	def forward(self, X, X_ctl=None):
		"""A forward pass of the model.

		This method takes in a nucleotide sequence X, a corresponding
		per-position value from a control track, and a per-locus value
		from the control track and makes predictions for the profile 
		and for the counts. This per-locus value is usually the
		log(sum(X_ctl_profile)+1) when the control is an experimental
		read track but can also be the output from another model.

		Parameters
		----------
		X: torch.tensor, shape=(batch_size, 4, length)
			The one-hot encoded batch of sequences.

		X_ctl: torch.tensor or None, shape=(batch_size, n_strands, length)
			A value representing the signal of the control at each position in 
			the sequence. If no controls, pass in None. Default is None.

		Returns
		-------
		y_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
			The output predictions for each strand trimmed to the output
			length.
		"""

		start, end = self.trimming, X.shape[2] - self.trimming

		X = self.irelu(self.iconv(X))
		for i in range(self.n_layers):
			X_conv = self.rrelus[i](self.rconvs[i](X))
			X = torch.add(X, X_conv)

		if X_ctl is None:
			X_w_ctl = X
		else:
			X_w_ctl = torch.cat([X, X_ctl], dim=1)

		y_profile = self.fconv(X_w_ctl)[:, :, start:end]

		# counts prediction
		X = torch.mean(X[:, :, start-37:end+37], dim=2)
		if X_ctl is not None:
			X_ctl = torch.sum(X_ctl[:, :, start-37:end+37], dim=(1, 2))
			X_ctl = X_ctl.unsqueeze(-1)
			X = torch.cat([X, torch.log(X_ctl+1)], dim=-1)

		y_counts = self.linear(X).reshape(X.shape[0], 1)
		return y_profile, y_counts


	def fit(self, training_data, optimizer, scheduler=None, X_valid=None, 
		X_ctl_valid=None, y_valid=None, max_epochs=100, batch_size=64, 
		dtype='float32', device='cuda', early_stopping=None):
		"""Fit the model to data and validate it periodically.

		This method controls the training of a BPNet model. It will fit the
		model to examples generated by the `training_data` DataLoader object
		and, if validation data is provided, will validate the model against 
		it at the end of each epoch and return those values.

		Two versions of the model will be saved: the best model found during
		training according to the validation measures, and the final model
		at the end of training. Additionally, a log will be saved of the
		training and validation statistics, e.g. time and performance.


		Parameters
		----------
		training_data: torch.utils.data.DataLoader
			A generator that produces examples to train on. If n_control_tracks
			is greater than 0, must product two inputs, otherwise must produce
			only one input.

		optimizer: torch.optim.Optimizer
			An optimizer to control the training of the model.

		scheduler: torch.optim.lr_scheduler, optional
			An optional learning rate scheduler which changes the learning rate
			across batches. If None, do not use a scheduler. Default is None.

		X_valid: torch.tensor or None, shape=(n, 4, 2114)
			A block of sequences to validate on periodically. If None, do not
			perform validation. Default is None.

		X_ctl_valid: torch.tensor or None, shape=(n, n_control_tracks, 2114)
			A block of control sequences to validate on periodically. If
			n_control_tracks is None, pass in None. Default is None.

		y_valid: torch.tensor or None, shape=(n, n_outputs, 1000)
			A block of signals to validate against. Must be provided if
			X_valid is also provided. Default is None.

		max_epochs: int
			The maximum number of epochs to train for, as measured by the
			number of times that `training_data` is exhausted. Default is 100.

		batch_size: int, optional
			The number of examples to include in each batch. Default is 64.
		
		dtype: str or torch.dtype
			The torch.dtype to use when training. Usually, this will be torch.float32
			or torch.bfloat16. Default is torch.float32.
		
		device: str
			The device to use for training and inference. Typically, this will be
			'cuda' but can be anything supported by torch. Default is 'cuda'.

		early_stopping: int or None, optional
			Whether to stop training early. If None, continue training until
			max_epochs is reached. If an integer, continue training until that
			number of epochs has been hit without improvement in performance. 
			Default is None.
		"""

		print("Warning: BPNet and ChromBPNet models trained using bpnet-lite may underperform those trained using the official repositories. See the GitHub README for further documentation.")
		
		if X_valid is not None:
			y_valid_counts = y_valid.sum(dim=2)

		if X_ctl_valid is not None:
			X_ctl_valid = (X_ctl_valid,)
			
		dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

		iteration = 0
		early_stop_count = 0
		best_loss = float("inf")
		self.logger.start()

		for epoch in range(max_epochs):
			tic = time.time()

			for data in training_data:
				X, y, labels = data[0], data[-2], data[-1]
				X_ctl = data[1].to(device) if len(data) == 4 else None
				
				X = X.to(device).float()
				y = y.to(device)

				# Clear the optimizer and set the model to training mode
				optimizer.zero_grad()
				self.train()

				# Make one training step
				with torch.autocast(device_type=device, dtype=dtype):
					y_hat_logits, y_hat_logcounts = self(X, X_ctl)
					
					training_profile_loss_, training_count_loss_, loss = _mixture_loss(y, 
						y_hat_logits, y_hat_logcounts, self.count_loss_weight, labels)

					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
					optimizer.step()

				iteration += 1

			train_time = time.time() - tic

			# Validate the model at the end of the epoch
			with torch.no_grad():
				self.eval()
				tic = time.time()

				y_hat_logits, y_hat_logcounts = predict(self, X_valid, args=X_ctl_valid, 
					batch_size=batch_size, dtype=dtype, device=device)

				valid_profile_loss, valid_count_loss, valid_loss = _mixture_loss(y_valid, 
					y_hat_logits, y_hat_logcounts, self.count_loss_weight)
				
				measures = calculate_performance_measures(y_hat_logits, 
					y_valid, y_hat_logcounts, kernel_sigma=7, 
					kernel_width=81, measures=['profile_pearson', 'count_pearson'])

				valid_profile_corr = numpy.nan_to_num(measures['profile_pearson'])
				valid_count_corr = numpy.nan_to_num(measures['count_pearson'])
				valid_time = time.time() - tic

				self.logger.add([epoch, 
					iteration, 
					train_time, 
					valid_time, 
					training_profile_loss_.item(), 
					training_count_loss_.item(), 
					valid_profile_loss.item(), 
					valid_profile_corr.mean(),
					valid_count_corr.mean(), 
					valid_count_loss.item(),
					(valid_loss < best_loss).item()])

				self.logger.save("{}.log".format(self.name))

				if valid_loss < best_loss:
					torch.save(self, "{}.torch".format(self.name))
					best_loss = valid_loss
					early_stop_count = -1
			
			if scheduler is not None:
				scheduler.step(valid_loss)

			early_stop_count += 1
			if early_stopping is not None and early_stop_count >= early_stopping:
				break

		torch.save(self, "{}.final.torch".format(self.name))


	@classmethod
	def from_chrombpnet_lite(cls, filename):
		"""Loads a model from ChromBPNet-lite TensorFlow format.
	
		This method will load a ChromBPNet-lite model from TensorFlow format.
		Note that this is not the same as ChromBPNet format. Specifically,
		ChromBPNet-lite was a preceeding package that had a slightly different
		saving format, whereas ChromBPNet is the packaged version of that
		code that is applied at scale.

		This method does not load the entire ChromBPNet model. If that is
		the desired behavior, see the `ChromBPNet` object and its associated
		loading functions. Instead, this loads a single BPNet model -- either
		the bias model or the accessibility model, depending on what is encoded
		in the stored file.


		Parameters
		----------
		filename: str
			The name of the h5 file that stores the trained model parameters.


		Returns
		-------
		model: BPNet
			A BPNet model compatible with this repository in PyTorch.
		"""

		h5 = h5py.File(filename, "r")
		w = h5['model_weights']

		if 'model_1' in w.keys():
			w = w['model_1']
			bias = False
		else:
			bias = True

		k, b = 'kernel:0', 'bias:0'
		name = "conv1d_{}_1" if not bias else "conv1d_{0}/conv1d_{0}"

		layer_names = []
		for layer_name in w.keys():
			try:
				idx = int(layer_name.split("_")[1])
				layer_names.append(idx)
			except:
				pass

		n_filters = w[name.format(1)][k].shape[2]
		n_layers = max(layer_names) - 2

		model = BPNet(n_layers=n_layers, n_filters=n_filters, n_outputs=1,
			n_control_tracks=0, trimming=None)

		convert_w = lambda x: torch.nn.Parameter(torch.tensor(
			x[:]).permute(2, 1, 0))
		convert_b = lambda x: torch.nn.Parameter(torch.tensor(x[:]))

		model.iconv.weight = convert_w(w[name.format(1)][k])
		model.iconv.bias = convert_b(w[name.format(1)][b])
		model.iconv.padding = 12

		for i in range(2, n_layers+2):
			model.rconvs[i-2].weight = convert_w(w[name.format(i)][k])
			model.rconvs[i-2].bias = convert_b(w[name.format(i)][b])

		model.fconv.weight = convert_w(w[name.format(n_layers+2)][k])
		model.fconv.bias = convert_b(w[name.format(n_layers+2)][b])
		model.fconv.padding = 12

		name = "logcounts_1" if not bias else "logcounts/logcounts"
		model.linear.weight = torch.nn.Parameter(torch.tensor(w[name][k][:].T))
		model.linear.bias = convert_b(w[name][b])
		return model


	@classmethod
	def from_chrombpnet(cls, filename):
		"""Loads a model from ChromBPNet TensorFlow format.
	
		This method will load one of the components of a ChromBPNet model
		from TensorFlow format. Note that a full ChromBPNet model is made up
		of an accessibility model and a bias model and that this will load
		one of the two. Use `ChromBPNet.from_chrombpnet` to end up with the
		entire ChromBPNet model.


		Parameters
		----------
		filename: str
			The name of the h5 file that stores the trained model parameters.


		Returns
		-------
		model: BPNet
			A BPNet model compatible with this repository in PyTorch.
		"""

		h5 = h5py.File(filename, "r")
		w = h5['model_weights']

		if 'bpnet_1conv' in w.keys():
			prefix = ""
		else:
			prefix = "wo_bias_"

		namer = lambda prefix, suffix: '{0}{1}/{0}{1}'.format(prefix, suffix)
		k, b = 'kernel:0', 'bias:0'

		n_layers = 0
		for layer_name in w.keys():
			try:
				idx = int(layer_name.split("_")[-1].replace("conv", ""))
				n_layers = max(n_layers, idx)
			except:
				pass

		name = namer(prefix, "bpnet_1conv")
		n_filters = w[name][k].shape[2]

		model = BPNet(n_layers=n_layers, n_filters=n_filters, n_outputs=1,
			n_control_tracks=0)

		convert_w = lambda x: torch.nn.Parameter(torch.tensor(
			x[:]).permute(2, 1, 0))
		convert_b = lambda x: torch.nn.Parameter(torch.tensor(x[:]))

		iname = namer(prefix, 'bpnet_1st_conv')

		model.iconv.weight = convert_w(w[iname][k])
		model.iconv.bias = convert_b(w[iname][b])
		model.iconv.padding = ((21 - 1) // 2,)

		for i in range(1, n_layers+1):
			lname = namer(prefix, 'bpnet_{}conv'.format(i))

			model.rconvs[i-1].weight = convert_w(w[lname][k])
			model.rconvs[i-1].bias = convert_b(w[lname][b])

		prefix = prefix + "bpnet_" if prefix != "" else ""

		fname = namer(prefix, 'prof_out_precrop')
		model.fconv.weight = convert_w(w[fname][k])
		model.fconv.bias = convert_b(w[fname][b])
		model.fconv.padding = ((75 - 1) // 2,)

		name = namer(prefix, "logcount_predictions")
		model.linear.weight = torch.nn.Parameter(torch.tensor(w[name][k][:].T))
		model.linear.bias = convert_b(w[name][b])
		return model
		

class BasePairNet(torch.nn.Module):
	"""A BPNet implementation matching that in basepairmodels

	This is a BPNet implementation that matches the one in basepairmodels and
	can be used to load models trained from that repository, e.g., those trained
	as part of the atlas project. The architecture of the model is identical to
	`BPNet` except that output from the residual layers is added to the 
	pre-activation outputs from the previous layer, rather than to the
	post-activation outputs from the previous layer. Additionally, the count
	prediction head takes the sum of the control track counts, adds two instead
	of one, and then takes the log. Neither detail dramatically changes
	performance of the model but is necessary to account for when loading
	trained models.
	

	Parameters
	----------
	n_filters: int, optional
		The number of filters to use per convolution. Default is 64.

	n_layers: int, optional
		The number of dilated residual layers to include in the model.
		Default is 8.

	n_outputs: int, optional
		The number of profile outputs from the model. Generally either 1 or 2 
		depending on if the data is unstranded or stranded. Default is 2.

	n_control_tracks: int, optional
		The number of control tracks to feed into the model. When predicting
		TFs, this is usually 2. When predicting accessibility, this is usualy
		0. When 0, this input is removed from the model. Default is 2.

	count_loss_weight: float, optional
		The weight to put on the count loss.

	profile_output_bias: bool, optional
		Whether to include a bias term in the final profile convolution.
		Removing this term can help with attribution stability and will usually
		not affect performance. Default is True.

	count_output_bias: bool, optional
		Whether to include a bias term in the linear layer used to predict
		counts. Removing this term can help with attribution stability but
		may affect performance. Default is True.

	name: str or None, optional
		The name to save the model to during training.

	trimming: int or None, optional
		The amount to trim from both sides of the input window to get the
		output window. This value is removed from both sides, so the total
		number of positions removed is 2*trimming.

	verbose: bool, optional
		Whether to display statistics during training. Setting this to False
		will still save the file at the end, but does not print anything to
		screen during training. Default is True.
	"""

	def __init__(self, n_filters=64, n_layers=8, n_outputs=2, 
		n_control_tracks=2, count_loss_weight=1, profile_output_bias=True, 
		count_output_bias=True, name=None, trimming=None, verbose=True):
		super(BasePairNet, self).__init__()
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_outputs = n_outputs
		self.n_control_tracks = n_control_tracks

		self.count_loss_weight = count_loss_weight
		self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)
		self.trimming = trimming or 2 ** n_layers

		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
		self.irelu = torch.nn.ReLU()

		self.rconvs = torch.nn.ModuleList([
			torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
				dilation=2**i) for i in range(1, self.n_layers+1)
		])
		self.rrelus = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(1, self.n_layers+1)
		])

		self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs, 
			kernel_size=75, padding=37, bias=profile_output_bias)
		
		n_count_control = 1 if n_control_tracks > 0 else 0
		self.linear = torch.nn.Linear(n_filters+n_count_control, 1, 
			bias=count_output_bias)

		self.logger = Logger(["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Training Count MSE", 
			"Validation MNLL", "Validation Profile Pearson", 
			"Validation Count Pearson", "Validation Count MSE", "Saved?"], 
			verbose=verbose)


	def forward(self, X, X_ctl=None):
		"""A forward pass of the model.

		This method takes in a nucleotide sequence X, a corresponding
		per-position value from a control track, and a per-locus value
		from the control track and makes predictions for the profile 
		and for the counts. This per-locus value is usually the
		log(sum(X_ctl_profile)+1) when the control is an experimental
		read track but can also be the output from another model.

		Parameters
		----------
		X: torch.tensor, shape=(batch_size, 4, length)
			The one-hot encoded batch of sequences.

		X_ctl: torch.tensor or None, shape=(batch_size, n_strands, length)
			A value representing the signal of the control at each position in 
			the sequence. If no controls, pass in None. Default is None.

		Returns
		-------
		y_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
			The output predictions for each strand trimmed to the output
			length.
		"""

		start, end = self.trimming, X.shape[2] - self.trimming
	
		X = self.iconv(X)
		for i in range(self.n_layers):
			X_a = self.rrelus[i](X)
			X_conv = self.rconvs[i](X_a)
			X = torch.add(X, X_conv)
		X = self.irelu(X)
	
		if X_ctl is None:
			X_w_ctl = X
		else:
			X_w_ctl = torch.cat([X, X_ctl], dim=1)
	
		y_profile = self.fconv(X_w_ctl)[:, :, start:end]
	
		# counts prediction
		X = torch.mean(X[:, :, start-37:end+37], dim=2)
		if X_ctl is not None:
			X_ctl = torch.sum(X_ctl[:, :, start:end], dim=(1, 2))
			X_ctl = X_ctl.unsqueeze(-1)
			X = torch.cat([X, torch.log(X_ctl+2)], dim=-1)
	
		y_counts = self.linear(X).reshape(X.shape[0], 1)
		return y_profile, y_counts


	@classmethod
	def from_bpnet(cls, filename):
		"""Loads a model from BPNet TensorFlow format.
	
		This method will allow you to load a BPNet model from the basepairmodels
		repo that has been saved in TensorFlow format. You do not need to have
		TensorFlow installed to use this function. The result will be a model
		whose predictions and attributions are identical to those produced when
		using the TensorFlow code.


		Parameters
		----------
		filename: str
			The name of the h5 file that stores the trained model parameters.


		Returns
		-------
		model: BPNet
			A BPNet model compatible with this repository in PyTorch.
		"""

		h5 = h5py.File(filename, "r")
		w, k, b = h5['model_weights'], 'kernel:0', 'bias:0'

		extract = lambda name, suffix: w['{0}/{0}/{1}'.format(name, suffix)][:]
		convert_w = lambda x: torch.nn.Parameter(torch.tensor(x).permute(2, 1, 
			0))
		convert_b = lambda x: torch.nn.Parameter(torch.tensor(x))

		n_layers, n_filters = 0, extract("main_conv_0", k).shape[2]
		for layer_name in w.keys():
			if 'main_dil_conv' in layer_name:
				n_layers = max(n_layers, int(layer_name.split("_")[-1]))

		model = cls(n_layers=n_layers, n_filters=n_filters, n_outputs=2,
			n_control_tracks=2, trimming=(2114-1000)//2)
		
		model.iconv.weight = convert_w(extract("main_conv_0", k))
		model.iconv.bias = convert_b(extract("main_conv_0", b))
		model.iconv.padding = ((model.iconv.weight.shape[-1] - 1) // 2,)

		for i in range(1, n_layers+1):
			lname = "main_dil_conv_{}".format(i)
			model.rconvs[i-1].weight = convert_w(extract(lname, k))
			model.rconvs[i-1].bias = convert_b(extract(lname, b))

		w0 = model.fconv.weight.numpy(force=True)
		wph = extract("main_profile_head", k)
		wpp = extract("profile_predictions", k)[0, :2]

		conv_weight = numpy.zeros_like(w0.transpose(2, 1, 0))
		conv_weight[:, :n_filters] = wph.dot(wpp) 
		conv_weight[37, n_filters:] = extract("profile_predictions", k)[0, 2:]
		model.fconv.weight = convert_w(conv_weight)
		model.fconv.bias = (convert_b(extract("main_profile_head", b) + 
			extract("profile_predictions", b)))
		model.fconv.padding = ((model.fconv.weight.shape[-1] - 1) // 2,)

		linear_weight = numpy.zeros_like(model.linear.weight.numpy(force=True))
		linear_weight[:, :n_filters] = (extract("main_counts_head", k).T * 
			extract("logcounts_predictions", k)[0])
		linear_weight[:, -1] = extract("logcounts_predictions", k)[1]
		
		model.linear.weight = convert_b(linear_weight)
		model.linear.bias = (convert_b(extract("main_counts_head", b) * 
			extract("logcounts_predictions", k)[0] + 
			extract("logcounts_predictions", b)))
		return model