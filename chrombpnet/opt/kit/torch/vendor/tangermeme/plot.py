# plot.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import warnings

from collections.abc import Callable
from typing import Any

import torch
import numpy
import pandas

import matplotlib
import matplotlib.collections as mc

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Bbox
from functools import lru_cache

from .deep_lift_shap import deep_lift_shap
from .utils import TangermemeWarning

import numpy as np


#plot_logo helper functions for placing annotations

def check_box_overlap(box1: Bbox, box2: Bbox) -> bool:
	"""Check if annotation label text boxes overlap."""
	return not(box1.x0>=box2.x1 or box2.x0>=box1.x1 or box1.y0>=box2.y1 or box2.y0>=box1.y1)


def check_box_overlap_bar(box1: Bbox, box2: Bbox) -> bool:
	"""Check if annotation bars overlap."""
	return not(box1.x0>=box2.x1 or box2.x0>=box1.x1 or box1.y0!=box2.y0)


def place_new_box(
	box: Bbox,
	box_list: list[Bbox],
	n_tracks: int = 4,
	show_extra: bool = True,
) -> tuple[Bbox, int]:
    """Place annotation text so that it does not overlap with existing text.

    The returned Bbox is a fresh object; the input `box` is not mutated.
    """
    box = Bbox.from_extents(box.x0, box.y0, box.x1, box.y1)
    box_height = box.y1 - box.y0
    box.y0 -= box_height
    box.y1 -= box_height

    if len(box_list)==0:
        return box,0

    overlap_exists = any([check_box_overlap(box,box2) for box2 in box_list])
    steps_down_taken = 0
    while overlap_exists:
        steps_down_taken += 1
        if steps_down_taken == n_tracks: #beyond the n_tracks limit: make boxes smaller
                if not show_extra:
                        box.y0 -= box_height
                        box.y1 -= box_height
                        return box, steps_down_taken
                box.y1-= box_height
                box.y0-= box_height/2
                box_height = box_height/2
                overlap_exists = any([check_box_overlap(box,box2) for box2 in box_list])
                continue

        box.y0 -= box_height
        box.y1 -= box_height
        overlap_exists = any([check_box_overlap(box,box2) for box2 in box_list])

    return box, steps_down_taken


def place_new_bar(
	box: Bbox,
	box_list: list[Bbox],
	y_step: float | None = None,
	n_tracks: int = 4,
	show_extra: bool = True,
) -> tuple[Bbox, int]:
    """
    Find a position for a new annotation bar such that it does not overlap with previously plotted bars.

    The returned Bbox is a fresh object; the input `box` is not mutated.
    """
    if y_step is None:
        raise ValueError("y_step must be provided.")

    box = Bbox.from_extents(box.x0, box.y0, box.x1, box.y1)

    if len(box_list)==0:
        return box, 0

    overlap_exists = any([check_box_overlap_bar(box,box2) for box2 in box_list])
    steps_down_taken = 0
    while overlap_exists:
        steps_down_taken += 1
        box.y0 -= y_step
        box.y1 -= y_step
        overlap_exists = any([check_box_overlap_bar(box,box2) for box2 in box_list])
    return box, steps_down_taken


@lru_cache(maxsize=4)
def _base_path(char):
    fp = FontProperties(family='monospace', weight='bold')
    tp = TextPath((0, 0), char, size=1, prop=fp)
    bbox = tp.get_extents()
    verts = (tp.vertices - [bbox.x0, bbox.y0]) / [bbox.width, bbox.height]
    return verts, tp.codes.copy()


def get_glyph_path(
	char: str,
	x: float,
	y: float,
	width: float,
	height: float,
	flip: bool = False,
) -> Path:
    verts, codes = _base_path(char)
    v = verts.copy()
    v[:, 0] = v[:, 0] * width + x
    if flip:
        v[:, 1] = (1 - v[:, 1]) * height + y
    else:
        v[:, 1] = v[:, 1] * height + y
    return Path(v, codes)


def plot_logo(
	X_attr: torch.Tensor | numpy.ndarray,
	ax: matplotlib.axes.Axes | None = None,
	color: str | dict | numpy.ndarray | list | None = None,
	color_cmap: str = "viridis",
	color_vmin: float | None = None,
	color_vmax: float | None = None,
	annotations: pandas.DataFrame | None = None,
	start: int | None = None,
	end: int | None = None,
	ylim: tuple[float, float] | None = None,
	spacing: float = 4,
	n_tracks: int = 4,
	alphabet: list[str] = ["A", "C", "G", "T"],
	min_height_pct: float = 0.02,
	score_key: str = 'score',
	show_extra: bool = True,
	show_score: bool = True,
	annot_cmap: str = "Set1",
) -> matplotlib.axes.Axes:
	"""Make a logo plot and optionally annotate it.

	This function will take in a matrix of weights for each character in a
	sequence and produce a plot where the characters have heights proportional
	to those weights. Attribution values from a predictive model are commonly
	used to weight the characters, but the weights can come from anywhere.

	Optionally, annotations can be provided in the form of a dataframe with
	contents described below in the parameters section. These annotations will
	be displayed underneath the characters in a manner that tries to avoid
	overlap across annotations.


	Parameters
	----------
	X_attr: torch.tensor, shape=(4, -1)
			A tensor of the attributions. Can be either the hypothetical
			attributions, where the entire matrix has values, or the projected
			attributions, where only the actual bases have their attributions
			stored, i.e., 3 values per column are zero.

	ax: matplotlib.pyplot.subplot or None, optional
			The art board to draw on. If None, choose the current artboard.

	color: str, dict, numpy.ndarray, list, or None, optional
			The coloring for the glyphs, which can be specified per character or
			per position depending on the form passed in:

					- None: color each character according to the standard coloring.
					- str: color every character with this single color.
					- dict: color each character according to a mapping from the
					  character to a color (e.g. ``{'A': 'red', 'C': 'blue', ...}``).
					- array-like: color each position independently. Must have the
					  same length as the last dimension of `X_attr` (before any
					  `start`/`end` slicing) and is sliced alongside `X_attr` under
					  the same conditions. May either be a 1D numeric vector, in
					  which case each value is mapped to a color through `color_cmap`
					  and the `color_vmin`/`color_vmax` bounds, or a sequence of
					  color specifications (names, hex strings, or an (length, 3)/
					  (length, 4) array of RGB(A) values), used verbatim. All glyphs
					  in a column share the color assigned to that position.

			When an array-like is passed but its length does not match the last
			dimension of `X_attr`, a `TangermemeWarning` is raised and the standard
			per-character coloring is used instead. Default is None.

	color_cmap: str, optional
			The colormap used to map a numeric `color` vector to colors. Ignored
			unless `color` is a 1D numeric array-like. Default is "viridis".

	color_vmin: float or None, optional
			The lower bound used to normalize a numeric `color` vector before
			mapping through `color_cmap`. If None, the minimum of the vector is
			used. Default is None.

	color_vmax: float or None, optional
			The upper bound used to normalize a numeric `color` vector before
			mapping through `color_cmap`. If None, the maximum of the vector is
			used. Default is None.

	annotations: pandas.DataFrame, optional
			A set of annotations with the following columns in any order except for
			`motif_name`, which can be called anything but must come first:

					- motif_name: the name of the motif
					- start: the start of the hit relative to the window provided
					- end: the end of the hit relative to the window provided
					- strand: the strand the hit is on (optional)
					- score: the score of the hit

			These will probably come from the output of the hit caller. Default is
			None.

	start: int or None, optional
			The start of the sequence to visualize. Must be non-negative and cannot
			be longer than the length of `X_attr`. If None, visualize the full
			sequence. Default is None.

	end: int or None, optional
			The end of the sequence to visualize. Must be non-negative and cannot be
			longer than the length of `X_attr`. If `start` is provided, `end` must 
			be larger. If None, visualize the full sequence. Default is None.

	ylim: tuple or None, optional
			The lower and upper bounds of the plot. Pass the bounds in here rather
			than setting them after calling this function if you want the annotation
			spacing to adjust to it. If None, use the default bounds. Default is
			None.

	n_tracks: int, optional
			The number of rows of annotation labels to plot with bars before simply
			putting the name of the motif. Default is 4.

	score_key: str, optional
			When annotations are provided, the name of the key to use as a score.
			Must have the semantics that a higher value means a "better" annotation.
			Default is 'score'.

	show_extra: bool, optional
			Whether to show motif names past the `n_tracks` number of rows. 
			If False, do not show those motifs. Default is True.

	show_score: bool, optional
			Whether to show the score of the hit. Sometimes, the annotation can take up
			too much space already and the scoring information is not as helpful, so
			disabling this will only display the motif hit name. Default is True.

	annot_cmap: str, list, or matplotlib.colors.ListedColormap, optional
			The colormap to use for the annotations. Rows of annotation labels receive 
			distinct colors, bars are colored according to their corresponding label.
			If a string, must be a valid matplotlib qualitative colormap name 
			If a list, must be a list of colors (list of colornames or list of RGB tuples).
			Labels are colorcoded if there is more than 1 row of annotations, otherwise black.


	Returns
	-------
	ax: plt.subplot
			A subplot that contains the plot.
	"""

	try:
			import matplotlib.pyplot as plt
	except ImportError:
			raise ImportError("Must install matplotlib before using.")

	###
	# Main glyph plotting code
	###

	if isinstance(X_attr, torch.Tensor):
		X_attr = X_attr.numpy(force=True)

	default_color = {
		'G': [1, .65, 0],
		'T': [1, 0, 0],
		'C': [0, 0, 1],
		'A': [0, .5, 0]
	}

	# `color` can be a per-character mapping (None/str/dict) or a per-position
	# array-like. Resolve which case we are in here; the per-position colors are
	# sliced alongside `X_attr` below and turned into RGBA after slicing.
	position_colors = None
	if color is None:
		color = default_color
	elif isinstance(color, str):
		color = {char: color for char in alphabet}
	elif isinstance(color, dict):
		pass # Use the dictionary
	else:
		if len(color) != X_attr.shape[-1]:
			warnings.warn("`color` is an array-like of length {} but `X_attr` "
				"has length {} along its last dimension; falling back to the "
				"default per-character coloring. Pass a dict to color by "
				"character or an array-like matching the sequence length to "
				"color by position.".format(len(color), X_attr.shape[-1]),
				TangermemeWarning)
			color = default_color
		else:
			position_colors = color
			color = default_color

	if start is not None and end is not None:
		X_attr = X_attr[:, start:end]
		if position_colors is not None:
			position_colors = position_colors[start:end]

	if ax is None:
		ax = plt.gca()

	if position_colors is not None:
		position_colors_ = numpy.asarray(position_colors)
		if position_colors_.ndim == 1 and position_colors_.dtype.kind in "fiu":
			vmin = color_vmin if color_vmin is not None \
				else position_colors_.min()
			vmax = color_vmax if color_vmax is not None \
				else position_colors_.max()
			norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
			cmap = plt.get_cmap(color_cmap)
			position_colors = cmap(norm(position_colors_))
		else:
			position_colors = matplotlib.colors.to_rgba_array(position_colors)

	paths, facecolors = [], []
	
	threshold = 0.0
	if min_height_pct is not None:
		threshold = np.abs(X_attr).max() * min_height_pct
	
	for pos in range(X_attr.shape[-1]):
		heights = X_attr[:, pos]
		order = np.argsort(heights)
		y_pos, y_neg = 0.0, 0.0
	
		for idx in order:
			h = heights[idx]
			if abs(h) < threshold:
				continue
	
			if position_colors is not None:
				facecolor = position_colors[pos]
			else:
				facecolor = color[alphabet[idx]]

			if h > 0:
				paths.append(get_glyph_path(alphabet[idx], pos, y_pos, 0.95, h))
				facecolors.append(facecolor)
				y_pos += h
			else:
				y_neg += h
				facecolors.append(facecolor)
				paths.append(get_glyph_path(alphabet[idx], pos, y_neg, 0.95, -h, flip=True))
	
	pos_total = np.sum(np.where(X_attr > 0, X_attr, 0), axis=0).max()
	neg_total = np.sum(np.where(X_attr < 0, X_attr, 0), axis=0).min()
	
	col = mc.PathCollection(paths, facecolors=facecolors, edgecolors='none', lw=0)
	ax.add_collection(col)
	ax.set_xlim(-0.1, X_attr.shape[-1])
	ax.set_ylim(neg_total, pos_total)
	ax.spines[['top', 'right', 'bottom']].set_visible(False)
	
	
	###
	# Handling potentially overlapping annotations
	###


	#Set annotation colormap
	if type(annot_cmap) == str:
		cmap = plt.get_cmap(annot_cmap)
	elif type(annot_cmap) == list:
		cmap = ListedColormap(annot_cmap)
	elif type(annot_cmap) == matplotlib.colors.ListedColormap:
		cmap = annot_cmap

	if annotations is not None:
		start, end = start or 0, end or X_attr.shape[-1]

		annotations_ = annotations[annotations['start'] > start]
		annotations_ = annotations_[annotations_['end'] < end]
		annotations_ = annotations_.sort_values(["start"], ascending=True)

		if len(annotations_) > 0:
			ylim = ylim or max(abs(X_attr.min()), abs(X_attr.max()))
			ax.set_ylim(-ylim, ylim)
			r = ylim*2
			y_offset_bars=ax.get_ylim()[0]/8
			y_offset_labels = 0

			#determine label text size and line width according to figure size
			bbox = ax.get_position()
			fig_width, fig_height = ax.get_figure().get_size_inches()
			width_in = bbox.width * fig_width
			height_in = bbox.height * fig_height
			labelsize=width_in*1.1
			linewidth = width_in*0.25

			#plotting annotation labels
			label_box_objects = []
			visible_label_box_objects = []
			label_text_boxes = []
			text_box_colors = []
			
			for i,(_, row) in enumerate(annotations_.iterrows()):
				motif = row.values[0]
				motif_start = int(row['start'])
				motif_end = int(row['end'])
				score = row[score_key]
				motif_start -= start
				motif_end -= start

				#define label text
				text = str(motif)
				if show_score:
						text += ' ({:3.3})'.format(score)
				text=text.replace(" ", "\n")

				#plot text box in top most position...
				text_box = ax.text(motif_start, y_offset_labels, text, fontsize=labelsize)
				ax.get_figure().canvas.draw()
				bbox = text_box.get_window_extent()

				#...shift box down if it overlaps with previously drawn boxes
				bbox_new, steps_down_taken = place_new_box(bbox, label_text_boxes, n_tracks=n_tracks,show_extra=show_extra)
				bbox_new_transformed = bbox_new.transformed(ax.transData.inverted())

				#color the text according to the number of downshifts
				text_color = cmap(steps_down_taken)
				if steps_down_taken >= n_tracks: #if box is beyond the n_tracks limit, plot in smaller font size and in light grey.
						text_color = (0.7,0.7,0.7)
						text_box.set_fontsize(labelsize/2)
						if not show_extra:
								text_color = (1,1,1,0)
				else:
						visible_label_box_objects.append(text_box)

				text_box.set_position((bbox_new_transformed.x0, bbox_new_transformed.y0))
				text_box.set_color(text_color)
				text_box_colors.append(text_color)
				label_text_boxes.append(bbox_new)
				label_box_objects.append(text_box)

			#plotting annotation bars
			bars_box_objects = []
			bars_boxes = []
			bars_ymins=[]
			for i,(_, row) in enumerate(annotations_.iterrows()):
				motif = row.values[0]
				motif_start = int(row['start'])
				motif_end = int(row['end'])
				score = row[score_key]
				motif_start -= start
				motif_end -= start

				bar_color = text_box_colors[i]
				if bar_color == (0.7,0.7,0.7) or bar_color == (1,1,1,0): #for labels beyond the n_tracks limit, no bar is plotted.
						continue

				xp = [motif_start-0.1, motif_end-0.5]
				yp = [y_offset_bars, y_offset_bars]

				#plot bar in topmost place...
				bar = ax.plot(xp, yp, color='0.3', linewidth=linewidth)
				ax.get_figure().canvas.draw()

				#...shift bar down if it overlaps with previously drawn bars
				bar_box = bar[0].get_window_extent()
				bar_box_new, steps_down_taken = place_new_bar(bar_box, bars_boxes, y_step=linewidth*2, n_tracks=n_tracks,show_extra=show_extra)
				bar_box_new_transformed = bar_box_new.transformed(ax.transData.inverted())
				bar[0].set_ydata([bar_box_new_transformed.y0, bar_box_new_transformed.y1])
				bars_boxes.append(bar_box_new)
				bar[0].set_color(bar_color)
				bars_ymins.append(bar_box_new_transformed.y0)
				bars_box_objects.append(bar[0])


			#shift text boxes down under the lowest bar
			bars_ymin = min(bars_ymins)
			for label_box in label_box_objects:
					label_box.set_y(label_box.get_position()[1] + bars_ymin) 

			#if there is only one row of annotations, set colors to black
			if len(set([vis_box.get_color() for vis_box in visible_label_box_objects])) == 1:
					for label_box in visible_label_box_objects:
							label_box.set_color((0,0,0))
					for bar_box in bars_box_objects:
							bar_box.set_color((0,0,0))

	return ax


_DEFAULT_TOOLTIP_CSS = """
.mpld3-tooltip {
	background: rgba(255, 255, 255, 0.95);
	border: 1px solid #cccccc;
	border-radius: 4px;
	padding: 6px 9px;
	font-family: sans-serif;
	font-size: 11px;
	color: #222222;
	box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
}
"""


# mpld3 draws its own axis domain lines and tick marks in a near-black
# `axiscolor` that ignores matplotlib spine visibility, so they cannot be
# removed from the Python side. This CSS hides the axis paths (the "spines")
# and tick marks while leaving the tick labels and the gridlines (which live
# under the separate `mpld3-xgrid` / `mpld3-ygrid` classes) untouched. The
# `!important` is needed to beat mpld3's id-prefixed rules regardless of order.
_DESPINE_CSS = """
.mpld3-xaxis path, .mpld3-yaxis path,
.mpld3-xaxis .tick line, .mpld3-yaxis .tick line {
	display: none !important;
}
"""


# mpld3 sets the grid color on the `<g class="tick">` wrapper, but the gridline
# `<line>` inside does not inherit it and renders black. Style the line elements
# directly (with `!important` to beat mpld3's own rules) to get a light grid.
_GRID_CSS = """
.mpld3-xgrid .tick line, .mpld3-ygrid .tick line {
	stroke: #c4c4c4 !important;
	stroke-width: 0.5px !important;
}
"""


# Columns whose tooltip label differs from the raw DataFrame column name. The
# seqlet caller emits bare `attribution` / `p-value` columns; spell them out as
# seqlet-specific so they are not confused with an annotation (e.g. TOMTOM)
# p-value carried in a separate column.
_TOOLTIP_COLUMN_LABELS = {
	'attribution': 'seqlet attribution',
	'p-value': 'seqlet p-value',
}


def _format_tooltip_value(value: Any) -> str:
	"""Format a single annotation value for display in an HTML tooltip.

	Floating-point values -- which include p-values that are frequently very
	small as well as attribution sums -- are rendered with `{:.3g}` so that
	both tiny and large magnitudes stay readable. Everything else is passed
	through `str`.
	"""

	if isinstance(value, (float, numpy.floating)):
		return "{:.3g}".format(value)
	return str(value)


def interactive_logo(
	X_attr: torch.Tensor | numpy.ndarray,
	ax: matplotlib.axes.Axes | None = None,
	color: str | dict | numpy.ndarray | list | None = None,
	color_cmap: str = "viridis",
	color_vmin: float | None = None,
	color_vmax: float | None = None,
	annotations: pandas.DataFrame | None = None,
	start: int | None = None,
	end: int | None = None,
	ylim: tuple[float, float] | None = None,
	alphabet: list[str] = ["A", "C", "G", "T"],
	min_height_pct: float = 0.02,
	box_alpha: float = 0.3,
	box_linewidth: float = 0,
	annot_cmap: str | list | ListedColormap = "Pastel1",
	label: bool = True,
	label_fontsize: float = 9,
	grid: bool = True,
	despine: bool = True,
	tooltip_css: str | None = None,
) -> matplotlib.axes.Axes:
	"""Make a logo plot with interactive, hoverable annotation boxes.

	This is an interactive counterpart to `plot_logo`. The sequence logo
	itself is drawn by reusing `plot_logo`, so the glyphs look identical.
	The difference is in how annotations are displayed: rather than the
	stacked underline-and-label tracks used by `plot_logo`, each annotation
	is drawn as a single translucent, pastel box spanning the full vertical
	extent of the plot across the positions of the hit. A label -- the `name`
	column when present, otherwise the annotation's index -- is written in
	small font in the upper-left corner of its box (unless `label=False`), and
	hovering over the box reveals a tooltip with the rest of the annotation's
	statistics.

	The reason for this idiom is that motif hits frequently carry several
	pieces of metadata worth inspecting together -- the seqlet p-value, the
	annotation (TOMTOM) p-value, the summed attribution, the strand, and so
	on -- which is more than a static label can show without cluttering the
	plot. Boxing the region and deferring the details to a hover tooltip keeps
	the logo readable while still exposing every column you pass in. Because
	the boxes are translucent, overlapping hits remain legible and are
	distinguished by their colors and corner labels rather than by being
	pushed onto separate tracks.

	Interactivity is provided by `mpld3`, which converts the matplotlib
	figure to D3-backed HTML. The tooltips therefore work both inline in a
	Jupyter notebook and in exported static HTML. `mpld3` is an optional
	dependency; this function imports it lazily and raises a helpful error if
	it is not installed. The figure returned has the tooltip plugin already
	attached, so to view it call `mpld3.display(ax.get_figure())` in a
	notebook (or `mpld3.enable_notebook()` before plotting), and to export it
	call `mpld3.fig_to_html(ax.get_figure())`.


	Parameters
	----------
	X_attr: torch.tensor, shape=(4, -1)
		A tensor of the attributions. Can be either the hypothetical
		attributions, where the entire matrix has values, or the projected
		attributions, where only the actual bases have their attributions
		stored, i.e., 3 values per column are zero.

	ax: matplotlib.axes.Axes or None, optional
		The art board to draw on. If None, choose the current artboard.
		Default is None.

	color: str, dict, numpy.ndarray, list, or None, optional
		The glyph coloring, passed through to `plot_logo`. Accepts the same
		forms: None for standard nucleotide coloring, a str or dict for
		per-character coloring, or an array-like for per-position coloring
		(see `plot_logo` for the full contract). Default is None.

	color_cmap: str, optional
		The colormap used to map a numeric per-position `color` vector to
		colors, passed through to `plot_logo`. Default is "viridis".

	color_vmin: float or None, optional
		The lower normalization bound for a numeric per-position `color`
		vector, passed through to `plot_logo`. If None, the minimum of the
		vector is used. Default is None.

	color_vmax: float or None, optional
		The upper normalization bound for a numeric per-position `color`
		vector, passed through to `plot_logo`. If None, the maximum of the
		vector is used. Default is None.

	annotations: pandas.DataFrame, optional
		A set of annotations to box and label. Must contain `start` and `end`
		columns giving each hit's position relative to the window; all other
		columns are shown in the hover tooltip. If a `name` column is present
		it is used as the box label and tooltip title; otherwise each box is
		labeled by its index among the annotations visible in the window. In
		the tooltip, `attribution` and `p-value` columns (as emitted by the
		seqlet caller) are relabeled `seqlet attribution` and `seqlet p-value`
		so they are not confused with a separate annotation p-value. Default
		is None.

	start: int or None, optional
		The start of the sequence to visualize. Must be non-negative and
		cannot be longer than the length of `X_attr`. If None, visualize the
		full sequence. Default is None.

	end: int or None, optional
		The end of the sequence to visualize. Must be non-negative and cannot
		be longer than the length of `X_attr`. If `start` is provided, `end`
		must be larger. If None, visualize the full sequence. Default is None.

	ylim: tuple or None, optional
		The lower and upper bounds of the plot as `(low, high)`. The
		annotation boxes span this full vertical range. If None, use the
		bounds chosen by `plot_logo`. Default is None.

	alphabet: list, optional
		A list of characters that comprise the alphabet. Default is
		['A', 'C', 'G', 'T'].

	min_height_pct: float, optional
		Passed through to `plot_logo`; glyphs shorter than this fraction of
		the tallest glyph are not drawn. Default is 0.02.

	box_alpha: float, optional
		The opacity of the annotation box fill, between 0 and 1. The corner
		label is drawn opaque regardless. Default is 0.3.

	box_linewidth: float, optional
		The width of the annotation box outline. When 0 (the default), no
		outline is drawn and the box is shown by its translucent fill alone;
		set to a positive value to draw a tinted border around each box.
		Default is 0.

	annot_cmap: str, list, or matplotlib.colors.ListedColormap, optional
		The colormap used to color the annotation boxes. Each annotation
		receives a distinct color. If a string, must be a valid matplotlib
		colormap name. If a list, must be a list of colors. Qualitative
		(listed) colormaps cycle by annotation index; continuous colormaps
		are sampled evenly across the annotations. Default is 'Pastel1'.

	label: bool, optional
		Whether to draw the name/index label in the corner of each box. The
		hover tooltip is unaffected by this. Default is True.

	label_fontsize: float, optional
		The font size of the label drawn in the corner of each box. Only
		applies when `label=True`. Default is 9.

	grid: bool, optional
		Whether to draw a light horizontal grid. mpld3 renders whatever grid
		matplotlib's style specifies, which is frequently a heavy default. When
		True, this overrides it with a subtle light-gray horizontal grid; when
		False, no grid is drawn at all (it does not fall back to the matplotlib
		default). Either way the appearance does not depend on the caller's
		rcParams. Default is True.

	despine: bool, optional
		Whether to remove the bottom and left axis lines. mpld3 draws its own
		near-black axis spines and tick marks that ignore matplotlib's spine
		settings; when True, these are hidden (via injected CSS) for a clean,
		despined look while the tick labels and grid remain. Default is True.

	tooltip_css: str or None, optional
		CSS string passed to the mpld3 tooltip plugin for styling the hover
		text. If None, a default stylesheet is used that gives the tooltip a
		semi-opaque white background, a border, and padding so the text is
		legible over the plot. Pass a string to override it entirely. Default
		is None.


	Returns
	-------
	ax: matplotlib.axes.Axes
		The axes on which the logo was drawn. The parent figure has the
		mpld3 tooltip plugin attached; render it with `mpld3.display` or
		`mpld3.fig_to_html`.
	"""

	try:
		from mpld3 import plugins
	except ImportError:
		raise ImportError("interactive_logo requires mpld3. Install it with "
			"`pip install mpld3` or `pip install tangermeme[interactive]`.")

	if ax is None:
		ax = plt.gca()

	# Draw the logo glyphs by reusing plot_logo without its annotation tracks.
	plot_logo(X_attr, ax=ax, color=color, color_cmap=color_cmap,
		color_vmin=color_vmin, color_vmax=color_vmax, start=start, end=end,
		alphabet=alphabet, min_height_pct=min_height_pct)

	if ylim is not None:
		ax.set_ylim(ylim)

	# mpld3 renders whatever grid the caller's matplotlib style specifies,
	# which is often a heavy default; set it explicitly for a consistent look.
	ax.grid(False)
	if grid:
		ax.set_axisbelow(True)
		ax.grid(True, axis='y', color="#c4c4c4", linewidth=0.5)

	# plot_logo already hides the top/right/bottom spines; hide the left one
	# too for the despined look. The mpld3-side axis lines are handled by CSS
	# injected alongside the tooltip below.
	if despine:
		ax.spines['left'].set_visible(False)

	if annotations is None:
		return ax

	view_start, view_end = start or 0, end or (
		X_attr.shape[-1] if isinstance(X_attr, numpy.ndarray) else X_attr.shape[-1])

	annotations_ = annotations[annotations['start'] > view_start]
	annotations_ = annotations_[annotations_['end'] < view_end]
	annotations_ = annotations_.sort_values(["start"], ascending=True)

	if len(annotations_) == 0:
		return ax

	if isinstance(annot_cmap, str):
		cmap = plt.get_cmap(annot_cmap)
	elif isinstance(annot_cmap, list):
		cmap = ListedColormap(annot_cmap)
	else:
		cmap = annot_cmap

	has_name = 'name' in annotations_.columns
	n = len(annotations_)

	y0, y1 = ax.get_ylim()
	height = y1 - y0

	rects, facecolors, edgecolors, labels = [], [], [], []
	for i, (_, row) in enumerate(annotations_.iterrows()):
		motif_start = int(row['start']) - view_start
		motif_end = int(row['end']) - view_start

		# Label by the `name` column when present, otherwise by the index among
		# the annotations visible in the window.
		label_text = str(row['name']) if has_name else str(i)

		if isinstance(cmap, ListedColormap):
			rgb = cmap(i % cmap.N)[:3]
		else:
			rgb = cmap(i / max(1, n - 1))[:3]

		x0 = motif_start - 0.1
		width = (motif_end - motif_start) + 0.05
		rects.append(Rectangle((x0, y0), width, height))
		facecolors.append((*rgb, box_alpha))
		# Darken the edge and label relative to the pastel fill so the box
		# outline and label stand out against the translucent interior.
		edgecolors.append((*tuple(0.7 * c for c in rgb), 1.0))

		if label:
			# Darker than the edge (0.7) so the on-top label reads clearly
			# against the pastel fill.
			text_color = tuple(0.4 * c for c in rgb)
			ax.text(x0 + 0.25, y1 - 0.05 * height, label_text,
				fontsize=label_fontsize, ha='left', va='top', color=text_color)

		parts = ["<b>{}</b>".format(label_text),
			"length: {:,}".format(motif_end - motif_start)]
		for col in annotations_.columns:
			if has_name and col == 'name':
				continue
			# Coordinates are raw counts -- show them with thousands separators
			# and never in scientific notation, unlike scores / p-values.
			if col in ('start', 'end'):
				value = "{:,}".format(int(row[col]))
			else:
				value = _format_tooltip_value(row[col])
			parts.append("{}: {}".format(
				_TOOLTIP_COLUMN_LABELS.get(col, col), value))
		labels.append("<br>".join(parts))

	# zorder 0.6 places the boxes behind the logo glyphs (default zorder 1) so
	# the letters render at full saturation, but above the grid (zorder 0.5).
	col = mc.PatchCollection(rects, facecolors=facecolors,
		edgecolors=edgecolors, linewidths=box_linewidth, zorder=0.6)
	ax.add_collection(col)

	css = tooltip_css if tooltip_css is not None else _DEFAULT_TOOLTIP_CSS
	if grid:
		css = css + _GRID_CSS
	if despine:
		css = css + _DESPINE_CSS
	tooltip = plugins.PointHTMLTooltip(col, labels=labels, css=css)
	plugins.connect(ax.get_figure(), tooltip)

	return ax


def plot_categorical_scatter(
	X: torch.Tensor,
	colors: dict | None = None,
	ax: matplotlib.axes.Axes | None = None,
	**kwargs: Any,
) -> matplotlib.axes.Axes:
	"""A scatterplot of category weights across a seq, useful for attributions.

	Frequently, when you calculate attributions you are considering a sequence
	that is too long to be easily visualized in the standard character format,
	e.g. in `plot_logo`. One could scan the sequence a few times to find the
	regions of high attribution but that is also time-consuming. This function
	creates a scatterplot where the coloring corresponds to the category (e.g.,
	which nucleotide is there) and the value is the weight encoded in the matrix.
	Basically, this is a way to consider attributions on an entire sequence and
	get a sense for which areas you may want to follow-up with.

	This function is not meant solely for attributions, though. You can put any
	value in the matrix to be visualized.


	Parameters
	----------
	X: torch.Tensor, shape=(alphabet_len, length)
		A sequence to be visualized.

	colors: list or None, optional
		The colors to use for each category. By default, uses the standard
		nucleotide color scheme.

	ax: matplotlib.axes.Axes or None, optional
		The axes to plot on. If None, use the current axes via `plt.gca()`.
		Default is None.

	**kwargs: any additional arguments


	Returns
	-------
	ax: matplotlib.axes.Axes
		The axes on which the scatterplot was drawn.
	"""

	if ax is None:
		ax = plt.gca()

	if colors is None:
		c = [[0, 0.5, 0], [0, 0, 1], [1, 0.65, 0], [1, 0, 0]]

	pos = numpy.arange(X.shape[-1])
	for i in range(X.shape[0]):
		idxs = torch.abs(X).argmax(axis=0) == i
		ax.scatter(pos[idxs], X.sum(axis=0)[idxs], c=[c[i]], **kwargs)

	return ax


def plot_attributions(
	models: torch.nn.Module | list[torch.nn.Module],
	X: torch.Tensor,
	func: Callable[..., Any] = deep_lift_shap,
	attribute_kwargs: dict | None = None,
	plot_kwargs: dict | None = None,
	layout: tuple[int, int] | None = None,
) -> tuple[matplotlib.figure.Figure, torch.Tensor]:
	"""A convenience function for calculating and then plotting attributions.
	
	This function will use one or more models and calculate attributions on one or
	more sequences. These attributions are then plotted according to the format in
	layout, and returned to the user. This is just a wrapped around using an
	attribution function and `plot_logo`, and so additional arguments can be 
	passed into the attribution or the plotting functions using the respective
	kwargs arguments. `start` and `end` are likely 
	to be common arguments to pass
	into `plot_kwargs`.
	
	There are three signatures supported. (1) A list of models are provided with
	the same length as the first dimension in `X`. The i-th model gets applied to
	the i-th element of X. This is the most flexible, as the same model or
	sequence could be passed in multiple times. (2) Rather than a list of models,
	a single model object is provided, and gets applied to all the elements of X.
	(3) Rather than multiple sequences, a single sequence is provided for X as a
	2D tensor. in which case all models are applied to it.
	
	By default, `deep_lift_shap` is used to calculate attributions but any
	function can be used as long as it has the same signature.


	Parameters
	----------
	models: torch.nn.Module or list of torch.nn.Modules
		One or a set of models to apply to the sequence/s.
	
	X: torch.Tensor, shape=(-1, len(alphabet), -1) or (len(alphabet), -1)
		One or more sequences to calculate attributions for and visualize.

	func: func, optional
		The attribution function to use. Default is deep_lift_shap.
	
	attribute_kwargs: dict or None, optional
		Arguments to pass into `func`. Default is None.
	
	plot_kwargs: dict or None, optional
		Arguments to pass into `plot_logo`. Default is None.

	layout: 2-tuple or None, optional
		A layout of the subplots, the first two numbers passed into
		`plt.subplot`. If None, assume (n, 1).

	Returns
	-------
	axs: list of plt.axis
		A list of artboards.
	
	attributions: torch.Tensor
		The attributions being visualized here. This is the full output from `func`,
		even when only a portion is visualized.
	"""

	try:
		import matplotlib.pyplot as plt
	except ImportError:
		raise ImportError("Must install matplotlib before using.")
	
	
	if isinstance(models, torch.nn.Module):
		if X.ndim != 3:
			raise ValueError("X must have 3 dimensions when only providing one model.")
		
		models = [models for _ in range(X.shape[0])]
		
	elif X.ndim == 2:
		X = X.repeat(len(models), 1, 1)
	
	else:
		if len(models) != len(X):
			raise ValueError("X must have first dimension equal to number of models.")
	
	if layout is None:
		layout = (len(X), 1)
	else:
		if layout[0] * layout[1] < len(X):
			raise ValueError("Layout must have >= number of models.")
			
	attribute_kwargs = attribute_kwargs or {}
	plot_kwargs = plot_kwargs or {}
	
	X_attrs = []
	for i, model in enumerate(models):
		X_attr = func(model, X[i:i+1], **attribute_kwargs)
		X_attrs.append(X_attr)
	X_attrs = torch.cat(X_attrs, dim=0).detach()
	
	axs = []
	for i in range(len(X_attrs)):
		ax = plt.subplot(*layout, i+1)
		axs.append(ax)
		
		plot_logo(X_attrs[i], ax=ax, **plot_kwargs)
	
	return axs, X_attrs

def plot_pwm(
	pwm: torch.Tensor | numpy.ndarray,
	ax: matplotlib.axes.Axes | None = None,
	name: str | None = None,
	alphabet: list[str] = ['A', 'C', 'G', 'T'],
	eps: float = 1e-7,
) -> matplotlib.axes.Axes:
	"""Plot an information-content weighted PWM as a sequence logo.

	This function takes in a PWM with shape `(len(alphabet), length)`,
	where each column (position) sums to 1 across the alphabet axis, and
	plots the information-content weighted version of it. To visualize
	the reverse complement, call this function a second time with
	`pwm[::-1, ::-1]`.

	The function no longer creates its own figure or calls `plt.show()`;
	it draws on the provided `ax` (or `plt.gca()` if none is given) and
	returns the axes used. This matches the convention used by
	`plot_logo` and `plot_categorical_scatter`.


	Parameters
	----------
	pwm: torch.Tensor or numpy.ndarray, shape=(len(alphabet), length)
		The PWM to visualize. Each *column* (i.e. position) must sum to 1
		across the alphabet axis.

	ax: matplotlib.axes.Axes or None, optional
		The axes to draw on. If None, use the current axes via
		`plt.gca()`. Default is None.

	name: str or None, optional
		The title for the plot. If None, no title is set. Default is None.

	alphabet: list, optional
		A list of characters that comprise the alphabet. Default is
		['A', 'C', 'G', 'T'].

	eps: float, optional
		A small pseudocount to add to counts to make the log work correctly.
		Default is 1e-7.


	Returns
	-------
	ax: matplotlib.axes.Axes
		The axes on which the logo was drawn.
	"""

	if isinstance(pwm, torch.Tensor):
		pwm = pwm.numpy(force=True)

	if ax is None:
		ax = plt.gca()

	bg = 0.25 * numpy.log(0.25) / numpy.log(2)

	ic = pwm * numpy.log(pwm + eps) / numpy.log(2) - bg
	ic = numpy.sum(ic, axis=0, keepdims=True)
	plot_logo(pwm * ic, ax=ax)

	if name is not None:
		ax.set_title(name)
	ax.set_xlabel("Motif Position")
	ax.set_ylabel("Information Content (Bits)", fontsize=10)

	return ax
