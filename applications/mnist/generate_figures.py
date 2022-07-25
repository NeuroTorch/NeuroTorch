import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import numpy as np

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

dict_param_name = {
	'hidden_layer_type': 'Dynamique',
	"use_recurrent_connection": "Connections récurrentes",
	"to_spikes_use_periods": 'Temps en période',
	"n_hidden_neurons": 'Taille de la couche cachée',
	"n_iteration": "Nombre d'itérations",
	"learn_beta": "Apprentissage de Beta",
	"n_steps": "Temps d'intégration",
}


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
		**kwargs):
	"""
	Plots a histogram of the given results.
	"""
	if list_col_names is None:
		list_col_names = [
			'hidden_layer_type',
			'use_recurrent_connection',
			'to_spikes_use_periods',
			'n_hidden_neurons',
			'learn_beta'
		]
	dataset_results = results[results['dataset_id'] == 'DatasetId.' + dataset]
	col_names = list(dict_param_name.keys())
	torem = list(set(col_names) - set(list_col_names))
	for elem in torem:
		col_names.remove(elem)
	# if hide_col is not None:
	# 	if type(hide_col) is list:
	# 		for col in hide_col:
	# 			col_names.remove(col)
	# 	else:
	# 		col_names.remove(hide_col)
	dataset_results = dataset_results.sort_values(
		by=col_names,
		ignore_index=True)
	if data_mask is not None:
		dataset_results = dataset_results[dataset_results[data_mask[0]] == data_mask[1]]

	y_data = dataset_results[y_axis] * 100
	y_data = y_data.tolist()
	xlabel = []
	dict_param_surname = dict(
		hidden_layer_type='',
		use_recurrent_connection='R ',
		to_spikes_use_periods='P ',
		n_hidden_neurons='H ',
		learn_beta='B ',
		n_steps='T ',
	)
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
					label += f'{surname}{"[✓]" if dataset_results.loc[i, col] == True else "[X]"}<br>'
				else:
					label += f'{surname}{str(dataset_results.loc[i, col]).split(".")[-1]}<br>'
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
	)
	)
	return figure


def plot_bar_result(
		results: pd.DataFrame,
		dataset_name: str,
		list_col_names: list,
		data_mask: tuple = None,
		list_col_names_xaxis=None,
):
	"""
	Plots a histogram of the given results.
	"""
	if list_col_names_xaxis is None:
		list_col_names_xaxis = [
			'hidden_layer_type',
			'use_recurrent_connection',
			'to_spikes_use_periods',
			'n_hidden_neurons',
			'learn_beta'
		]
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


def make_data_for_box_plot(results: pd.DataFrame, dataset_name: str, ydata: str) -> dict:
	"""
	Returns the data for the box plot.
	"""
	dataset_results = results[results['dataset_id'] == 'DatasetId.' + dataset_name]
	y_data = dataset_results[ydata] * 100
	box_plot_data = {}
	for param in dict_param_name.keys():
		if param not in dataset_results.columns:
			continue
		if len(dataset_results[param].unique()) <= 1:
			continue
		for param_value in dataset_results[param].unique():
			if param == 'hidden_layer_type':
				param_value_name = param_value.split('.')[-1]
			elif param == 'use_recurrent_connection':
				param_value_name = 'REC [✓]' if param_value else 'REC [X]'
			elif param == 'to_spikes_use_periods':
				param_value_name = 'P [✓]' if param_value else 'P [X]'
			elif param == 'n_hidden_neurons':
				param_value_name = f'HN {param_value}'
			elif param == 'n_steps':
				param_value_name = f'T {param_value}'
			else:
				param_value_name = param_value
			try:
				box_plot_data[f'{param_value_name}'] = y_data[dataset_results[param] == param_value].tolist()
			except ValueError as err:
				print(f'{param} {param_value} {err}')
				continue
	return box_plot_data


def box_plot_accuracy(results: pd.DataFrame, dataset_name: str):
	box_plot_data = make_data_for_box_plot(results, dataset_name, 'test_accuracy')
	figure = go.Figure()
	palette = sns.color_palette("tab10", 999)
	Legendg = ['Dynamique', 'Recurrence', 'Période', 'N neurones']
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
				# legendgroup=f'{Legendg[i // 2]}',
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
		title=dict(text='Accuracy [%]'),
		# range=[0, 100],
	)
	return figure


if __name__ == '__main__':
	result = load_results('tr_data/results.csv')
	best_result = result.sort_values(by='test_accuracy', ascending=False).iloc[0]
	print(f"f{best_result = }")
	# for _dataset_name_ in ['MNIST', 'FASHION_MNIST']:
	# 	box_plot_accuracy(result, _dataset_name_).show()
	# 	plot_bar_result(
	# 		result,
	# 		_dataset_name_,
	# 		[
	# 			'test_accuracy',
	# 			'test_f1',
	# 			# 'test_precision',
	# 			# 'test_recall'
	# 		],
	# 		list_col_names_xaxis=[
	# 			'hidden_layer_type',
	# 			'use_recurrent_connection',
	# 			'n_steps',
	# 		]
	# 	).show()



