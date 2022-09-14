import os
from functools import partial
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

import neurotorch as nt
from applications.figure_generation_util import load_results, format_table, box_plot_on_metric, \
	metric_per_all_variable
from neurotorch.modules import LayerType

plot_layout = dict(
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


def gen_predictor_figures(
		filename: str,
):
	figures_folder = os.path.join(os.path.dirname(filename), 'figures')
	os.makedirs(figures_folder, exist_ok=True)
	dict_param_name = {
		'n_time_steps'            : 'Number of time steps [-]',
		'n_encoder_steps'         : 'Number of encoder steps [-]',
		'n_units'                 : 'Number of units [-]',
		"encoder_type"            : "Encoder type [-]",
		"use_recurrent_connection": "Recurrent Connection [-]",
		"optimizer"               : 'Optimizer [-]',
		"learning_rate"           : 'Learning rate [-]',
		"dt"                      : "Time step [s]",
		"smoothing_sigma"         : "Smoothing sigma [-]",
		"reg"                     : "Regularization [-]",
		"hh_init"                 : "HH init [-]",
		"learn_decoder"           : "Learn decoder [-]",
		"decoder_alpha_as_vec"    : "Decoder alpha as vec [-]",
		"seed"                    : "Seed [-]",
		"training_time"           : "Training time [s]",
		"pVar"                    : "pVar [-]",
		"pVar_chunks"                : "pVar chunks [-]",
	}
	dict_param_surname = dict(
		n_time_steps='T',
		n_encoder_steps='TE',
		n_units='N',
		use_recurrent_connection='R',
		encoder_type='E',
		optimizer='O',
		learning_rate='Lr',
		dt='dt',
		smoothing_sigma='$\sigma$',
		# training_time='Trt',
		seed='Seed',
	)
	result = load_results(filename).sort_values(by='pVar', ascending=False)
	best_result = result.iloc[:3]
	result_sort_by_chunks = load_results(filename).sort_values(by='pVar_chunks', ascending=False)
	best_result_sort_by_chunks = result_sort_by_chunks.iloc[:3]
	cols_oi = [
		'n_time_steps',
		'n_encoder_steps',
		'n_units',
		'encoder_type',
		'use_recurrent_connection',
		"optimizer",
		'dt',
		'smoothing_sigma',
		# 'training_time',
		'pVar',
		'pVar_chunks',
	]
	metrics_cols = ["pVar", "pVar_chunks", "training_time"]
	value_rename = {
		str(nt.SpyLIFLayer): 'SpyLIF',
		str(nt.SpyALIFLayer): 'SpyALIF',
		str(nt.ALIFLayer)  : 'ALIF',
		str(nt.LIFLayer)   : 'LIF',
	}
	non_unique_dict_params = {
		k: v
		for k, v in dict_param_name.items()
		if result[k].nunique() > 1
	}
	non_unique_cols = list(set(list(non_unique_dict_params.keys()) + metrics_cols))
	filtered_result = format_table(
		result,
		cols=non_unique_cols,
		metrics_to_maximize=['pVar', 'pVar_chunks'],
		metrics_to_minimize=['training_time'],
		cols_rename=dict_param_name,
		value_rename=value_rename,
	)
	filtered_best_result = format_table(
		best_result,
		cols=non_unique_cols,
		metrics_to_maximize=['pVar', 'pVar_chunks'],
		metrics_to_minimize=['training_time'],
		cols_rename=dict_param_name,
		value_rename=value_rename,
	)
	filtered_best_result_sort_by_chunks = format_table(
		best_result_sort_by_chunks,
		cols=non_unique_cols,
		metrics_to_maximize=['pVar', 'pVar_chunks'],
		metrics_to_minimize=['training_time'],
		cols_rename=dict_param_name,
		value_rename=value_rename,
	)
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(f"result:\n{filtered_result.to_latex(index=False, escape=False)}")
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(f"best_result:\n{filtered_best_result.to_latex(index=False, escape=False)}")
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(f"best_result_sort_by_chunks:\n{filtered_best_result_sort_by_chunks.to_latex(index=False, escape=False)}")
	for dataset_name in [
		'timeSeries_2020_12_16_cr3_df.npy'
	]:
		# box_plot_on_metric(
		# 	result, 'pVar',
		# 	dataset_name=dataset_name,
		# 	dict_param_name=dict_param_name,
		# 	dict_param_surname=dict_param_surname,
		# 	value_rename=value_rename,
		# 	plot_layout=plot_layout,
		# ).show()
		temp_dict_params = {
			k: v
			for k, v in dict_param_name.items()
			if result[result["dataset_name"] == dataset_name][k].nunique() > 1 and k not in metrics_cols
		}
		fig, axes = plt.subplots(nrows=len(temp_dict_params), ncols=3, figsize=(20, 10))
		axes[0, 0].set_title("pVar [-]")
		metric_per_all_variable(
			result, 'pVar',
			dataset_name=dataset_name,
			dict_param_name=temp_dict_params,
			value_rename=value_rename,
			metric_rename="",
			fig=fig, axes=axes[:, 0],
			# filename=os.path.join(figures_folder, f"{dataset_name.split('.')[0]}_pVar_per_all_variable.png"),
			show=False,
		)
		axes[0, 1].set_title("pVar chunks [-]")
		metric_per_all_variable(
			result, 'pVar_chunks',
			dataset_name=dataset_name,
			dict_param_name=temp_dict_params,
			value_rename=value_rename,
			metric_rename="",
			fig=fig, axes=axes[:, 1],
			# filename=os.path.join(figures_folder, f"{dataset_name.split('.')[0]}_pVar_per_all_variable.png"),
			show=False,
		)
		axes[0, 2].set_title("Training time [s]")
		metric_per_all_variable(
			result, 'training_time',
			dataset_name=dataset_name,
			dict_param_name=temp_dict_params,
			value_rename=value_rename,
			metric_rename='',
			fig=fig, axes=axes[:, 2],
			filename=os.path.join(figures_folder, f"{dataset_name.split('.')[0]}_metrics_per_all_variable.png"),
			show=True,
		)
		

def gen_autoencoder_figures(filename: str):
	figures_folder = os.path.join(os.path.dirname(filename), 'figures')
	os.makedirs(figures_folder, exist_ok=True)
	dict_param_name = {
		'n_encoder_steps'         : 'Number of encoder steps',
		'n_units'                 : 'Number of units',
		"encoder_type"            : "Encoder type",
		"use_recurrent_connection": "Recurrent Connection",
		"dt"                      : "Time step",
		"optimizer"               : 'Optimizer',
		# "seed"                    : "Seed",
	}
	dict_param_surname = dict(
		n_encoder_steps='TE',
		n_units='N',
		encoder_type='E',
		use_recurrent_connection='R',
		learning_rate='Lr',
		dt='dt',
		optimizer='O',
		seed='Seed',
	)
	value_rename = {
		str(nt.SpyLIFLayer): 'SpyLIF',
		str(nt.ALIFLayer): 'ALIF',
		str(nt.LIFLayer): 'LIF',
	}
	result = load_results(filename).sort_values(by='pVar', ascending=False)
	
	best_result = result.iloc[:3]
	cols_oi = [
		'n_encoder_steps',
		'n_units',
		'encoder_type',
		'use_recurrent_connection',
		'dt',
		'optimizer',
		'pVar',
	]
	filtered_result = format_table(
		result,
		cols=cols_oi,
		metrics_to_maximize=['pVar'],
		cols_rename=dict_param_name,
		value_rename=value_rename,
	)
	filtered_best_result = format_table(
		best_result,
		cols=cols_oi,
		metrics_to_maximize=['pVar'],
		cols_rename=dict_param_name,
		value_rename=value_rename,
	)
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(f"result:\n{filtered_result.to_latex(index=False, escape=False)}")
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(f"best_result:\n{filtered_best_result.to_latex(index=False, escape=False)}")
	for dataset_name in [
		'timeSeries_2020_12_16_cr3_df.npy'
	]:
		box_plot_on_metric(
			result, 'pVar',
			dataset_name=dataset_name,
			dict_param_name=dict_param_name,
			dict_param_surname=dict_param_surname,
			value_rename=value_rename,
			plot_layout=plot_layout,
		).show()
		temp_dict_params = {
			k: v
			for k, v in dict_param_name.items()
			if result[result["dataset_name"] == dataset_name][k].nunique() > 1
		}
		metric_per_all_variable(
			result, 'pVar',
			dataset_name=dataset_name,
			dict_param_name=temp_dict_params,
			value_rename=value_rename,
			filename=os.path.join(figures_folder, f"{dataset_name.split('.')[0]}_pVar_per_all_variable.png"),
			show=True,
		)


if __name__ == '__main__':
	# gen_autoencoder_figures('spikes_autoencoder_checkpoints_004/results.csv')
	gen_predictor_figures('predictor_checkpoints_004/results.csv')
	



