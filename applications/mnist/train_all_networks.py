from applications.mnist.results_generation import get_training_params_space, train_all_params

if __name__ == '__main__':
	df = train_all_params(
		training_params=get_training_params_space(),
		n_iterations=100,
		data_folder="tr_data",
		verbose=False
	)
	print(df)

