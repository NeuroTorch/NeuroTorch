import pprint

import numpy as np
import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from applications.fit_wilson_cowan_with_lif.results_generation import train_with_params
from neurotorch.modules.layers import LayerType
from neurotorch.transforms.spikes_encoders import LIFEncoder
from neurotorch.utils import set_seed

if __name__ == '__main__':
	logs_file_setup(__file__, add_stdout=False)
	log_device_setup(deepLib=DeepLib.Pytorch)
	torch.autograd.set_detect_anomaly(True)
	torch.cuda.set_per_process_memory_fraction(0.5)
	
	set_seed(42)
	
	n_units = 16
	t_0 = np.random.rand(n_units)
	forward_weights = 3 * np.random.randn(n_units, n_units)
	mu = np.random.randn(n_units, )
	r = np.random.rand(1).item()
	tau = 1.0
	
	results = train_with_params(
		{
			"n_steps": 10,
			"n_encoder_steps": 8,
			"n_units": n_units,
			"dt": 2e-2,
			"optimizer": "Adam",
			"learning_rate": 1e-2,
			"encoder": LIFEncoder,
			"hidden_layer_type": LayerType.SpyLIF,
			"readout_layer_type": LayerType.SpyLI,
		},
		t_0=t_0,
		forward_weights=forward_weights,
		mu=mu,
		r=r,
		tau=tau,
		n_iterations=100,
		verbose=True,
		show_training=False,
		force_overwrite=True,
		data_folder="tr_test",
	)
	pprint.pprint(results, indent=4)
	results["history"].plot(show=True)
