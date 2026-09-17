# marginalize.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pandas
import seaborn

from bpnetlite.bpnet import _ProfileLogitScaling
from bpnetlite.bpnet import CountWrapper
from bpnetlite.bpnet import ProfileWrapper
from bpnetlite.chrombpnet import _Log, _Exp

from tangermeme.io import read_meme
from tangermeme.io import one_hot_encode

from tangermeme.marginalize import marginalize
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.deep_lift_shap import _nonlinear

from tangermeme.plot import plot_logo

from tqdm import tqdm

import matplotlib.pyplot as plt


def path_to_image_html(path):
	return '<img src="' + path + '" width="240" >'


def _plot_profiles(y, ylim, color, path, figsize=(10,3)):
	plt.figure(figsize=figsize)
	plt.plot(y, color=color)

	seaborn.despine()
	plt.xlim(0, y.shape[0])
	plt.xlabel("Relative Coordinate", fontsize=12)
	plt.ylabel("Predicted Signal", fontsize=12)
	plt.ylim(*ylim)
	plt.yticks(fontsize=12)
	plt.savefig(path)
	plt.close()


def _plot_counts(y_before, y_after, xlim, ylim, color, path, figsize=(10,3)):
	zmax = max(xlim[1], ylim[1])
	zmin = min(xlim[0], ylim[0])

	plt.figure(figsize=figsize)
	plt.plot([zmin, zmax], [zmin, zmax], color='0.5')
	plt.scatter(y_before, y_after, color=color, s=5)

	seaborn.despine()
	plt.xlabel("Pred Counts Before", fontsize=12)
	plt.ylabel("Pred Counts After", fontsize=12)
	plt.xlim(*xlim)
	plt.ylim(*ylim)
	plt.yticks(fontsize=12)
	plt.savefig(path)
	plt.close()


def _plot_attributions(y, ylim, path, figsize=(10,3), **kwargs):
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111)

	plot_logo(y.T, ax=ax)
	
	plt.yticks(fontsize=12)
	plt.ylim(*ylim)
	plt.savefig(path)
	plt.close()


def marginalization_report(model, motifs, X, output_dir, batch_size=64,  
	attributions=True, minimal=False, device='cuda', verbose=False):
	"""Create an HTML report showing the impact of each motif.

	Take in a predictive model, a MEME file of motifs, and a set of sequences,
	and run a marginalization experiment on each motif. Store the images
	to the output directory, and then create an HTML report that puts together
	these images.


	Parameters
	----------
	model: bpnetlite.bpnet.BPNet or bpnetlite.chrombpnet.ChromBPNet
		A BPNet- or ChromBPNet-style model that outputs predictions for a
		profile head and a count head.

	motif: str
		A motif to insert into the middle of the background sequences.

	X: torch.Tensor, shape=(n, 4, 2114)
		A one-hot encoded set of n sequences to run through the model.

	output_dir: str
		The folder name to put all the images that are generated.

	batch_size: int, optional
		The number of examples to run at a time in each batch. Default is 64.

	attributions: bool, optional
		Whether to calculate attributions as well as calculating predictions
		when doing marginalizations. Because calculating attributions is the
		most time-intensive aspect, setting this parameter to False can save
		time. Default is True

	minimal: bool, optional
		Whether to produce a minimal report, which shows the differences in
		outputs, or the full report, which shows the results before and after
		insertion as well as the differences. Potentially useful for debugging.
		Default is False.
	
	device: str, optional
		The device to run operations on. Usually is either 'cuda' or 'cpu'.
		Default is 'cuda'.

	verbose: bool, optional
		Whether to print a progress bar as motifs are marginalized over. Default
		is False.
	"""

	motifs = list(read_meme(motifs).items())

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	results = {
		'name': [],
		'sequence': [],
		'profile (before)': [],
		'profile (after)': [],
		'profile (diff)': [],
		'counts': [],
		'profile attributions (before)': [],
		'profile attributions (after)': [],
		'profile attributions (diff)': [],
		'counts attributions (before)': [],
		'counts attributions (after)': [],
		'counts attributions (diff)': []
	}


	p, c, a = 'profile', 'counts', ' attributions'
	pb, pa, pd = '.profile.before', '.profile.after', '.profile.diff'
	pab, paa, pad = '.pattr.before', '.pattr.after', '.pattr.diff'
	cab, caa, cad = '.cattr.before', '.cattr.after', '.cattr.diff'

	mid, w = X.shape[-1] // 2, 15
	s, e = mid - w, mid + w

	p_before, p_after, p_diff = [], [], []
	c_before, c_after, c_diff = [], [], []
	ap_before, ap_after, ap_diff = [], [], []
	ac_before, ac_after, ac_diff = [], [], []

	for i, (name, pwm) in tqdm(enumerate(motifs), disable=not verbose):
		motif = ''.join(numpy.array(['A', 'C', 'G', 'T'])[pwm.argmax(axis=0)])

		(y_profile_before, y_counts_before), (y_profile_after, 
			y_counts_after) = marginalize(model, X, motif, 
			batch_size=batch_size, device=device)

		y_profile_before = torch.nn.functional.softmax(y_profile_before, dim=-1)
		y_profile_after = torch.nn.functional.softmax(y_profile_after, dim=-1)

		p_before.append(y_profile_before.mean(axis=0).T)
		p_after.append(y_profile_after.mean(axis=0).T)
		p_diff_ = (y_profile_after - y_profile_before).mean(axis=0).T
		p_diff.append(p_diff_)

		c_before.append(y_counts_before)
		c_after.append(y_counts_after)

		if attributions:
			Xp_attr_before, Xp_attr_after = marginalize(ProfileWrapper(
				model.double()), X.double(), motif, func=deep_lift_shap, 
				additional_nonlinear_ops={
					_ProfileLogitScaling: _nonlinear,
					_Log: _nonlinear,
					_Exp: _nonlinear
				}, 
				n_shuffles=1, batch_size=batch_size, device=device)

			Xc_attr_before, Xc_attr_after = marginalize(
				CountWrapper(model).double(), X.double(), motif, 
				func=deep_lift_shap, n_shuffles=1,
				additional_nonlinear_ops={
					_ProfileLogitScaling: _nonlinear,
					_Log: _nonlinear,
					_Exp: _nonlinear
				}, 
				batch_size=batch_size,
				device=device)

			ap_before.append(Xp_attr_before.mean(axis=0)[:, s:e].T)
			ap_after.append(Xp_attr_after.mean(axis=0)[:, s:e].T)
			ap_diff_ = (Xp_attr_after - Xp_attr_before).mean(axis=0)[:, s:e].T
			ap_diff.append(ap_diff_)

			ac_before.append(Xc_attr_before.mean(axis=0)[:, s:e].T)
			ac_after.append(Xc_attr_after.mean(axis=0)[:, s:e].T)
			ac_diff_ = (Xc_attr_after - Xc_attr_before).mean(axis=0)[:, s:e].T
			ac_diff.append(ac_diff_)

	multistack = lambda args: [torch.stack(a).numpy() for a in args]
	p_before, p_after, p_diff = multistack([p_before, p_after, p_diff])
	c_before, c_after = multistack([c_before, c_after])

	multilims = lambda args: [(x.min() * 0.95, x.max() * 1.05) for x in args] 
	pb_lim, pa_lim, pd_lim = multilims([p_before, p_after, p_diff])
	cb_lim, ca_lim = multilims([c_before, c_after])

	if attributions:
		apb, apa, apd = multistack([ap_before, ap_after, ap_diff])
		acb, aca, acd = multistack([ac_before, ac_after, ac_diff])

		apb_lim, apa_lim, apd_lim = multilims([apb, apa, apd])
		acb_lim, aca_lim, acd_lim = multilims([acb, aca, acd])


	idxs = (c_after - c_before).mean(axis=1)[:, 0].argsort()[::-1]
	for i, idx in enumerate(idxs):
		name, pwm = motifs[idx]
		name = name.replace(" ", "_").replace("/", "-")
		oname = output_dir + name
		motif = ''.join(numpy.array(['A', 'C', 'G', 'T'])[pwm.argmax(axis=0)])

		if not minimal:
			_plot_profiles(p_before[idx], pb_lim, color='0.5', 
				path=oname + pb + ".png")
			_plot_profiles(p_after[idx], pa_lim, color='0.5', 
				path=oname + pa + ".png")

		_plot_profiles(p_diff[idx], pd_lim, color='c', 
			path=oname + pd + ".png")
		_plot_counts(c_before[idx], c_after[idx], cb_lim, ca_lim, color='m', 
			path=oname + ".counts.png")

		if attributions:
			if not minimal:
				_plot_attributions(apb[idx], apb_lim, oname + pab + ".png")
				_plot_attributions(apa[idx], apa_lim, oname + paa + ".png")

				_plot_attributions(acb[idx], acb_lim, oname + cab + ".png")
				_plot_attributions(aca[idx], aca_lim, oname + caa + ".png")

			_plot_attributions(apd[idx], apd_lim, oname + pad + ".png")
			_plot_attributions(acd[idx], acd_lim, oname + cad + ".png")

		motif_ = motif[:25] + ('...' if len(motif) > 25 else '')

		results['name'].append(name)
		results['sequence'].append(motif_)
		results[p + ' (before)'].append(name + pb + ".png")
		results[p + ' (after)'].append(name + pa + ".png")
		results[p + ' (diff)'].append(name + pd + ".png")
		results[c].append(name + ".counts.png")

		if attributions:
			results[p + a + ' (before)'].append(name + pab + ".png")
			results[p + a + ' (after)'].append(name + paa + ".png")
			results[p + a + ' (diff)'].append(name + pad + ".png")
			results[c + a + ' (before)'].append(name + cab + ".png")
			results[c + a + ' (after)'].append(name + caa + ".png")
			results[c + a + ' (diff)'].append(name + cad + ".png")

	if not attributions:
		del results[p + a + ' (before)']
		del results[p + a + ' (after)']
		del results[p + a + ' (diff)']
		del results[c + a + ' (before)']
		del results[c + a + ' (after)']
		del results[c + a + ' (diff)']

	formatters = {name: path_to_image_html for name in results.keys() 
		if name not in ('name', 'sequence')}

	results_df = pandas.DataFrame(results)
	if minimal:
		if attributions:
			results_df = results_df[['name', 'sequence', 'profile (diff)', 
				'counts', 'profile attributions (diff)', 
				'counts attributions (diff)']]
		else:
			results_df = results_df[['name', 'sequence', 'profile (diff)', 
				'counts']]
			
	results_df.to_html(open('{}/marginalization.html'.format(output_dir), 'w'),
		escape=False, formatters=formatters, index=False)
