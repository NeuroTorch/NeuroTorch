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
		# "training_time"           : "Training time [s]",
		"seed"                    : "Seed [-]",
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
	]
	value_rename = {
		str(nt.SpyLIFLayer): 'SpyLIF',
		str(nt.ALIFLayer)  : 'ALIF',
		str(nt.LIFLayer)   : 'LIF',
	}
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
			if result[result["dataset_name"] == dataset_name][k].nunique() > 1
		}
		fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
		metric_per_all_variable(
			result, 'pVar',
			dataset_name=dataset_name,
			dict_param_name=temp_dict_params,
			value_rename=value_rename,
			metric_rename="pVar [-]",
			fig=fig, axes=axes[0],
			# filename=os.path.join(figures_folder, f"{dataset_name.split('.')[0]}_pVar_per_all_variable.png"),
			# show=True,
		)
		metric_per_all_variable(
			result, 'training_time',
			dataset_name=dataset_name,
			dict_param_name=temp_dict_params,
			value_rename=value_rename,
			metric_rename='Training time [s]',
			fig=fig, axes=axes[1],
			# filename=os.path.join(figures_folder, f"{dataset_name.split('.')[0]}_tr_time_per_all_variable.png"),
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
	# gen_autoencoder_figures('spikes_autoencoder_checkpoints_002/results.csv')
	gen_predictor_figures('predictor_checkpoints_pVar_vs_time_steps/results.csv')
	



