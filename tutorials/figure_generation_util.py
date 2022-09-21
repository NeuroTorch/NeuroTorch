from functools import partial
from typing import Dict, List, Optional, Iterable, Union

import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from neurotorch.modules import LayerType


default_plot_layout = dict(
	paper_bgcolor='rgb(243, 243, 243)',
	plot_bgcolor='rgb(243, 243, 243)',
	legend=dict(font_size=35, borderwidth=3, ),
	xaxis=dict(
		tickfont_size=32,
		zeroline=False,
		showgrid=False,
		title_font_size=24,
		showline=True,
		linewidth=4,
		linecolor='black',
	),
	yaxis=dict(
		showgrid=False,
		tickfont_size=42,
		zeroline=False,
		title_font_size=40,
		showline=True,
		linewidth=4,
		linecolor='black',
	)
)


def load_results(file_path='tr_data/results.csv') -> pd.DataFrame:
	"""
	Loads the results from the given file path.
	"""
	return pd.read_csv(file_path)


def _plot_bar_result(
		figure: go.Figure,
		results: pd.DataFrame,
		dataset: str,
		y_axis: str,
		data_mask: tuple = None,
		list_col_names=None,
		dict_param_name: Optional[Dict[str, str]] = None,
		dict_param_surname: Optional[Dict[str, str]] = None,
		y_axis_multiplier: Optional[float] = 1.0,
		**kwargs
) -> go.Figure:
	"""
	Plots a histogram of the given results.
	
	:param figure: The figure to plot the results on.
	:param results: The results to plot.
	:param dataset: The dataset name used to filter the results.
	:param y_axis: The y axis to plot e.g. the metric.
	:param data_mask: The data mask to use to filter the results.
	:param list_col_names: The list of column names to use to filter the results.
	:param dict_param_name: The dictionary of parameter names to format the table.
	:param dict_param_surname: The dictionary of parameter surnames to format the x axis.
	:param y_axis_multiplier: The multiplier to use to format the y axis.
	:param kwargs: Additional keyword arguments to pass to the plotly.graph_objects.Bar constructor.
	:return: The figure with the results plotted.
	"""
	if list_col_names is None:
		list_col_names = results.columns
	if dict_param_name is None:
		dict_param_name = {}
	if dict_param_surname is None:
		raise ValueError('dict_param_surname must be provided.')
	dataset_results = results[results['dataset_name'] == dataset]
	col_names = list(dict_param_name.keys())
	torem = list(set(col_names) - set(list_col_names))
	for elem in torem:
		col_names.remove(elem)
	dataset_results = dataset_results.sort_values(
		by=col_names,
		ignore_index=True
	)
	if data_mask is not None:
		dataset_results = dataset_results[dataset_results[data_mask[0]] == data_mask[1]]

	y_data = dataset_results[y_axis] * y_axis_multiplier
	y_data = y_data.tolist()
	xlabel = []
	for i in range(dataset_results.shape[0]):
		label = ''
		for col in list_col_names:
			try:
				surname = dict_param_surname[col]
			except KeyError:
				print(f'{col} is not a valid parameter name.')
				continue
			try:
				if dataset_results.loc[i, col] in [True, False]:
					label += f'{surname}: {"[✓]" if dataset_results.loc[i, col] == True else "[X]"}<br>'
				else:
					label += f'{surname}: {str(dataset_results.loc[i, col]).split(".")[-1]}<br>'
			except KeyError:
				print(f'{col} not in dataset_results[{i}|{col}]')
				continue
		xlabel.append(label)

	figure.add_trace(go.Bar(
		y=y_data,
		x=xlabel,
		name=y_axis,
		text=list(map(lambda a: round(a, 2), y_data)),
		textposition='auto',
		textangle=90,
		textfont_size=32,
		**kwargs
	))
	return figure


def plot_bar_result(
		results: pd.DataFrame,
		dataset_name: str,
		list_col_names: list,
		data_mask: tuple = None,
		list_col_names_xaxis=None,
		plot_layout: Optional[Dict] = None,
) -> go.Figure:
	"""
	Plots a histogram of the given results.
	
	:param results: The results to plot.
	:param dataset_name: The dataset name used to filter the results.
	:param list_col_names: The list of column names to use to filter the results.
	:param data_mask: The data mask to use to filter the results.
	:param list_col_names_xaxis: The list of column names to use to format the x axis.
	:param plot_layout: The layout to use to format the plot.
	:return: The figure with the results plotted.
	"""
	if list_col_names_xaxis is None:
		list_col_names_xaxis = results.columns
	if plot_layout is None:
		plot_layout = default_plot_layout
	figure = go.Figure()
	palette = sns.color_palette("rocket", len(list_col_names))
	for i, y_axis in enumerate(list_col_names):
		color = f'rgb{tuple(map(lambda a: int(a * 255), palette[i]))}'
		figure = _plot_bar_result(
			figure,
			results,
			dataset_name,
			y_axis,
			data_mask,
			list_col_names_xaxis,
			marker_color=color
		)
	figure.update_layout(plot_layout)
	figure.update_layout(
		barmode='group',
		legend=dict(
			x=1.01,
			y=1.0,
			xanchor='left',
			yanchor='top',
			borderwidth=3,
		),
		# xaxis_tickangle=70,
		uniformtext=dict(mode="hide", minsize=18),
	)
	figure.update_xaxes(
		ticks="outside",
		tickwidth=4,
		tickcolor='black'
	)
	figure.update_yaxes(
		title=dict(text='Performance [%]'),
		range=[0, 100],
	)
	return figure


def make_data_for_box_plot(
		results: pd.DataFrame,
		ydata: str,
		*,
		dataset_name: Optional[str] = None,
		dict_param_name: Optional[Dict[str, str]] = None,
		dict_param_surname: Optional[Dict[str, str]] = None,
		value_rename: Optional[Dict[str, str]] = None,
) -> dict:
	"""
	Returns the data for the box plot.
	
	:param results: The results to plot.
	:param dataset_name: The dataset name used to filter the results.
	:param ydata: The y data to use to filter the results e.g. the performance.
	:param dict_param_name: The dictionary of parameter names to filter and format the x axis.
	:param dict_param_surname: The dictionary of parameter surnames to format the x axis.
	:param value_rename: The dictionary of parameter values to format the x axis.
	:return: The data for the box plot.
	"""
	if dict_param_name is None:
		dict_param_name = {}
	if dict_param_surname is None:
		dict_param_surname = {}
	if value_rename is None:
		value_rename = {}
	if dataset_name is None:
		dataset_results = results
	else:
		dataset_results = results[results['dataset_name'] == dataset_name]
	y_data = dataset_results[ydata]
	box_plot_data = {}
	for param in dict_param_name.keys():
		if param not in dataset_results.columns:
			continue
		if len(dataset_results[param].unique()) <= 1:
			continue
		for param_value in dataset_results[param].unique():
			surname = dict_param_surname.get(param, '')
			value_surmame = value_rename.get(param_value, param_value)
			if isinstance(param_value, bool):
				param_value_name = f"{surname}[✓]" if param_value else f"{surname}[X]"
			elif isinstance(param_value, (str, LayerType)) and LayerType.from_str(param_value) is not None:
				param_value_name = f"{surname}: {LayerType.from_str(param_value).name}"
			else:
				param_value_name = f"{surname}: {value_surmame}"
			try:
				box_plot_data[f'{param_value_name}'] = y_data[dataset_results[param] == param_value].tolist()
			except ValueError as err:
				print(f'{param} {param_value} {err}')
				continue
	return box_plot_data


def box_plot_on_metric(
		results: pd.DataFrame,
		metric: str,
		*,
		metric_unit: str = '-',
		dataset_name: Optional[str] = None,
		dict_param_name: Optional[Dict[str, str]] = None,
		dict_param_surname: Optional[Dict[str, str]] = None,
		value_rename: Optional[Dict[str, str]] = None,
		plot_layout: Optional[Dict] = None,
) -> go.Figure:
	"""
	Plots a box plot on the given metric.
	
	:param results: The results to plot.
	:param dataset_name: The dataset name used to filter the results.
	:param metric: The metric to plot on.
	:param metric_unit: The metric unit to use to format the y axis.
	:param dict_param_name: The dictionary of parameter names to filter and format the x axis.
	:param dict_param_surname: The dictionary of parameter surnames to format the x axis.
	:param value_rename: The dictionary of parameter values to format the x axis.
	:param plot_layout: The layout to use to format the plot.
	:return: The figure with the box plot.
	"""
	if plot_layout is None:
		plot_layout = default_plot_layout
	box_plot_data = make_data_for_box_plot(
		results, metric,
		dataset_name=dataset_name,
		dict_param_name=dict_param_name,
		dict_param_surname=dict_param_surname,
		value_rename=value_rename,
	)
	figure = go.Figure()
	palette = sns.color_palette("tab10", 999)
	for i, (name, values) in enumerate(box_plot_data.items()):
		color = f'rgb{tuple(map(lambda a: int(a * 255), palette[i]))}'
		figure.add_trace(
			go.Box(
				y=values,
				name=name,
				boxpoints='all',
				pointpos=0,
				marker=dict(
					color=color,
					size=12
				),
			)
		)
	figure.update_layout(plot_layout)
	figure.update_xaxes(
		ticks="outside",
		tickwidth=4,
		tickcolor='black',
		tickfont_size=38,

	)
	figure.update_yaxes(
		title=dict(text=f'{metric} [{metric_unit}]'),
		# range=[0, 100],
	)
	return figure


def format_table(
		df: pd.DataFrame,
		*,
		cols: Optional[List[str]] = None,
		metrics_to_maximize: Optional[List[str]] = None,
		metrics_to_minimize: Optional[List[str]] = None,
		cols_to_percent: Optional[List[str]] = None,
		cols_rename: Optional[Dict[str, str]] = None,
		n_digits: int = 3,
		checkmark_as_latex_command: bool = True,
		value_rename: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
	formatted_table = df.copy()
	if metrics_to_maximize is None:
		metrics_to_maximize = []
	if metrics_to_minimize is None:
		metrics_to_minimize = []
	if cols_to_percent is None:
		cols_to_percent = []
	if cols is None:
		cols = formatted_table.columns
	formatted_table = formatted_table[cols]
	# for layer_type in LayerType:
	# 	formatted_table = formatted_table.replace(f"{layer_type}", layer_type.name, regex=True)
	checkmark_symbol = '$\\checkmark$' if checkmark_as_latex_command else '✓'
	formatted_table = formatted_table.replace(True, checkmark_symbol, regex=True)
	formatted_table = formatted_table.replace(False, "x", regex=True)
	for col in metrics_to_maximize+metrics_to_minimize:
		if col in cols_to_percent:
			formatted_table[col] = formatted_table[col] * 100
		formatted_table[col] = formatted_table[col].round(n_digits)
		if col in metrics_to_maximize:
			opt_value = formatted_table[col].max()
		elif col in metrics_to_minimize:
			opt_value = formatted_table[col].min()
		else:
			raise ValueError(f'{col} is not in {metrics_to_maximize} or {metrics_to_minimize}')
		formatted_table[col] = formatted_table[col].astype(str)
		formatted_table[col] = formatted_table[col].replace(f'{opt_value}', f"\\textbf{{{opt_value}}}", regex=True)
	if value_rename is not None:
		for key, val in value_rename.items():
			formatted_table = formatted_table.replace(key, val)
	if cols_rename is not None:
		formatted_table.rename(columns=cols_rename, inplace=True)
	formatted_table = formatted_table.rename(
		{c_name: c_name.replace('_', ' ') for c_name in formatted_table.columns},
		axis='columns'
	)
	return formatted_table


def format_table_metric_value(
		x,
		bold_value: float = np.inf,
		n_digits: int = 2,
):
	x = round(100*x, n_digits)
	if np.isclose(x, bold_value):
		x = f"\\textbf{{{x}}}"
	return x


def metric_per_all_variable(
		results: pd.DataFrame,
		metric: str,
		*,
		dataset_name: Optional[str] = None,
		dict_param_name: Optional[Dict[str, str]] = None,
		value_rename: Optional[Dict[str, str]] = None,
		metric_rename: Optional[str] = None,
		fig: Optional[plt.Figure] = None,
		axes: Optional[Union[plt.Axes, Iterable[plt.Axes]]] = None,
		filename: Optional[str] = None,
		show: bool = False,
):
	"""
	Returns the data for the box plot.

	:param results: The results to plot.
	:param dataset_name: The dataset name used to filter the results.
	:param metric: The y data to use to filter the results e.g. the performance.
	:param dict_param_name: The dictionary of parameter names to filter and format the x axis.
	:param value_rename: The dictionary of parameter values to format the x axis.
	:param metric_rename: The metric name to use to format the y axis.
	:param filename: The filename to save the plot.
	:param show: Whether to show the plot.
	:return: None
	"""
	if dataset_name is None:
		dataset_results = results
	else:
		dataset_results = results[results['dataset_name'] == dataset_name]
	if dict_param_name is None:
		columns = list(set(dataset_results.columns) - {'dataset_name', metric})
		dict_param_name = {
			c_name: c_name.replace('_', ' ')
			for c_name in columns
		}
	
	if value_rename is None:
		value_rename = {}
	if metric_rename is None:
		metric_rename = dict_param_name.pop(metric, metric.replace('_', ' '))
	y_data = dataset_results[metric]
	columns = list(dict_param_name.keys())
	
	nrows = int(np.sqrt(len(columns)))
	ncols = int(np.ceil(len(columns) / nrows))
	if fig is None:
		assert axes is None, "If fig is None, axes must be None"
		fig, axes_view = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*3))
	else:
		assert axes is not None, "If fig is not None, axes must be not None"
		if isinstance(axes, plt.Axes):
			axes_view = [axes]
		else:
			axes_view = axes
		axes_view = np.asarray(axes_view)
		assert axes_view.size == len(columns), "axes must have the same length as columns"
	axes_view = np.ravel(axes)
	index = 0
	for i, col in enumerate(columns):
		index = i
		xticks_labels = np.asarray([x for x in dataset_results[col].unique()])
		xticks_labels_renamed = np.asarray([value_rename.get(x, x) for x in xticks_labels])
		indexes_sort = np.argsort(xticks_labels_renamed)
		xticks_labels_renamed = xticks_labels_renamed[indexes_sort]
		xticks_labels = xticks_labels[indexes_sort]
		xticks = np.arange(len(xticks_labels))
		x_data = np.asarray([
			np.where(np.isclose(xticks_labels, x))
			if isinstance(x, float) else
			np.where(xticks_labels == x)
			for x in dataset_results[col]
		]).squeeze()
		y_mean = np.asarray([y_data[dataset_results[col] == x].mean() for x in xticks_labels])
		y_std = np.asarray([y_data[dataset_results[col] == x].std() for x in xticks_labels])
		axes_view[i].plot(xticks, y_mean, '-', color='black', label='mean')
		axes_view[i].fill_between(xticks, y_mean - y_std, y_mean + y_std, color='black', alpha=0.2, label='std')
		axes_view[i].plot(x_data, y_data, '.')
		axes_view[i].set_xlabel(dict_param_name.get(col, col))
		axes_view[i].set_ylabel(metric_rename)
		axes_view[i].set_xticks(xticks)
		axes_view[i].set_xticklabels(xticks_labels_renamed)
		axes_view[i].legend()
	
	for ax in axes_view[index+1:]:
		ax.set_visible(False)
	
	fig.set_tight_layout(True)
	if filename is not None:
		fig.savefig(filename)
	if show:
		plt.show()
	return fig, axes

	








