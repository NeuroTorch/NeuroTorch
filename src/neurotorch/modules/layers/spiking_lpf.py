from typing import Optional, Tuple

import numpy as np
import torch
from . import SpyLIFLayer

from ...dimension import SizeTypes


class SpyLIFLayerLPF(SpyLIFLayer):
	"""
	The SpyLIF dynamics is a more complex variant of the LIF dynamics (class :class:`LIFLayer`) allowing it to have a
	greater power of expression. This variant is also inspired by Neftci :cite:t:`neftci_surrogate_2019` and also
	contains  two differential equations like the SpyLI dynamics :class:`SpyLI`. The equation :eq:`SpyLIF_I` presents
	the synaptic current update equation with euler integration while the equation :eq:`SpyLIF_V` presents the
	synaptic potential update.

	In this version (LPF), the spikes are filtered with a low pass filter (LPF) described by the equation
	:eq:`lpf`.

	.. math::
		:label: SpyLIF_I

		\\begin{equation}
			I_{\\text{syn}, j}^{t+\\Delta t} = \\alpha I_{\text{syn}, j}^{t} + \\sum_{i}^{N} W_{ij}^{\\text{rec}} z_i^t
			+ \\sum_i^{N} W_{ij}^{\\text{in}} x_i^{t+\\Delta t}
		\\end{equation}


	.. math::
		:label: SpyLIF_V

		\\begin{equation}
			V_j^{t+\\Delta t} = \\left(\\beta V_j^t + I_{\\text{syn}, j}^{t+\\Delta t}\\right) \\left(1 - z_j^t\\right)
		\\end{equation}


	.. math::
		:label: spylif_alpha

		\\begin{equation}
			\\alpha = e^{-\\frac{\\Delta t}{\\tau_{\\text{syn}}}}
		\\end{equation}

	with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

	.. math::
		:label: spylif_beta

		\\begin{equation}
			\\beta = e^{-\\frac{\\Delta t}{\\tau_{\\text{mem}}}}
		\\end{equation}

	with :math:`\\tau_{\\text{syn}}` being the decay time constant of the synaptic current.

	The output of neuron :math:`j` at time :math:`t` denoted :math:`z_j^t` is defined by the equation :eq:`spylif_z` .

	.. math::
		:label: spylif_z

		z_j^t = H(V_j^t - V_{\\text{th}})

	where :math:`V_{\\text{th}}` denotes the activation threshold of the neuron and the function :math:`H(\\cdot)`
	is the Heaviside function defined as :math:`H(x) = 1` if :math:`x \\geq 0` and :math:`H(x) = 0` otherwise.

	.. math::
		:label: lpf

		\\mathcal{F}_{\\text{lpf}-\\alpha}(z_j^t) = {\\text{lpf}-\\alpha} \\mathcal{F}_\\alpha(z_j^{t-1}) + z_j^t

	SpyTorch library: https://github.com/surrogate-gradient-learning/spytorch.

	The variables of the equations :eq:`SpyLIF_I` and :eq:`SpyLIF_V` are described by the following definitions:

		- :math:`N` is the number of neurons in the layer.
		- :math:`I_{\\text{syn}, j}^{t}` is the synaptic current of neuron :math:`j` at time :math:`t`.
		- :math:`V_j^t` is the synaptic potential of the neuron :math:`j` at time :math:`t`.
		- :math:`\\Delta t` is the integration time step.
		- :math:`z_j^t` is the spike of the neuron :math:`j` at time :math:`t`.
		- :math:`\\alpha` is the decay constant of the synaptic current over time (equation :eq:`spylif_alpha`).
		- :math:`\\beta` is the decay constant of the membrane potential over time (equation :eq:`spylif_beta`).
		- :math:`W_{ij}^{\\text{rec}}` is the recurrent weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`W_{ij}^{\\text{in}}` is the input weight of the neuron :math:`i` to the neuron :math:`j`.
		- :math:`x_i^{t}` is the input of the neuron :math:`i` at time :math:`t`.

	:Attributes:
		- :attr:`alpha` (torch.nn.Parameter): Decay constant of the synaptic current over time (equation :eq:`spylif_alpha`).
		- :attr:`beta` (torch.nn.Parameter): Decay constant of the membrane potential over time (equation :eq:`spylif_beta`).
		- :attr:`threshold` (torch.nn.Parameter): Activation threshold of the neuron (:math:`V_{\\text{th}}`).
		- :attr:`gamma` (torch.nn.Parameter): Slope of the Heaviside function (:math:`\\gamma`).
		- :attr:`lpf_alpha` (float): Decay constant of the low pass filter over time (equation :eq:`lpf`).
	"""
	
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			use_recurrent_connection: bool = True,
			use_rec_eye_mask: bool = False,
			dt: float = 1e-3,
			device: Optional[torch.device] = None,
			**kwargs
	):
		"""
		Constructor for the SpyLIFLayerLPF layer.

		:param input_size: The size of the input.
		:type input_size: Optional[SizeTypes]
		:param output_size: The size of the output.
		:type output_size: Optional[SizeTypes]
		:param name: The name of the layer.
		:type name: Optional[str]
		:param use_recurrent_connection: Whether to use the recurrent connection.
		:type use_recurrent_connection: bool
		:param use_rec_eye_mask: Whether to use the recurrent eye mask.
		:type use_rec_eye_mask: bool
		:param spike_func: The spike function to use.
		:type spike_func: Callable[[torch.Tensor], torch.Tensor]
		:param learning_type: The learning type to use.
		:type learning_type: LearningType
		:param dt: Time step (Euler's discretisation).
		:type dt: float
		:param device: The device to use.
		:type device: Optional[torch.device]
		:param kwargs: The keyword arguments for the layer.

		:keyword float tau_syn: The synaptic time constant :math:`\\tau_{\\text{syn}}`. Default: 5.0 * dt.
		:keyword float tau_mem: The membrane time constant :math:`\\tau_{\\text{mem}}`. Default: 10.0 * dt.
		:keyword float threshold: The threshold potential :math:`V_{\\text{th}}`. Default: 1.0.
		:keyword float gamma: The multiplier of the derivative of the spike function :math:`\\gamma`. Default: 100.0.
		:keyword float spikes_regularization_factor: The regularization factor for the spikes. Higher this factor is,
			the more the network will tend to spike less. Default: 0.0.
		:keyword float lpf_alpha: The decay constant of the low pass filter over time (equation :eq:`lpf`).
			Default: np.exp(-dt / tau_mem).

		"""
		super(SpyLIFLayerLPF, self).__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			dt=dt,
			device=device,
			**kwargs
		)
		self.lpf_alpha = self.kwargs["lpf_alpha"]
	
	def _set_default_kwargs(self):
		super()._set_default_kwargs()
		self.kwargs.setdefault("lpf_alpha", np.exp(-self.dt / self.kwargs["tau_mem"]))
	
	def create_empty_state(
			self,
			batch_size: int = 1,
			**kwargs
	) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			([membrane potential of shape (batch_size, self.output_size)],
			[synaptic current of shape (batch_size, self.output_size)],
			[low pass filtered spikes of shape (batch_size, self.output_size)],
			[spikes of shape (batch_size, self.output_size)])

		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		kwargs["n_hh"] = 3
		V, I, Z = super(SpyLIFLayerLPF, self).create_empty_state(batch_size=batch_size, **kwargs)
		Z_filtered = Z.clone()
		kwargs["n_hh"] = 4
		return tuple([V, I, Z_filtered, Z])
	
	def forward(
			self,
			inputs: torch.Tensor,
			state: Tuple[torch.Tensor, ...] = None,
			**kwargs
	):
		assert inputs.ndim == 2, f"Inputs must be of shape (batch_size, input_size), got {inputs.shape}."
		batch_size, nb_features = inputs.shape
		V, I_syn, z_filtered, Z = self._init_forward_state(state, batch_size, inputs=inputs)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		next_I_syn = self.alpha * I_syn + input_current + rec_current
		next_V = (self.beta * V + next_I_syn) * (1.0 - Z.detach())
		next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)
		
		next_z_filtered = self.lpf_alpha * z_filtered + next_Z
		return next_z_filtered, (next_V, next_I_syn, next_z_filtered, next_Z)
	
	def extra_repr(self) -> str:
		_repr = super().extra_repr()
		_repr += f", low_pass_filter_alpha={self._low_pass_filter_alpha:.2f}"
		return _repr


