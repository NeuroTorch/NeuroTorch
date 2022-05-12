import enum
from typing import Tuple, Type

import numpy as np
import torch
from torch import nn

from . import HeavisideSigmoidApprox, SpikeFunction


class LayerType(enum.Enum):
	LIF = 0
	ALIF = 1
	Izhikevich = 2
	LI = 3


class RNNLayer(torch.nn.Module):
	def __init__(
			self,
			input_size: int,
			output_size: int,
			use_recurrent_connection=True,
			use_rec_eye_mask=True,
			dt=1e-3,
			device=None,
			**kwargs
	):
		super(RNNLayer, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.use_recurrent_connection = use_recurrent_connection
		self.device = device
		if self.device is None:
			self._set_default_device_()

		self.dt = dt
		self.kwargs = kwargs
		self._set_default_kwargs()

		self.forward_weights = nn.Parameter(
			torch.empty((self.input_size, self.output_size), device=self.device, dtype=torch.float32),
			requires_grad=True
		)
		self.use_rec_eye_mask = use_rec_eye_mask
		if use_recurrent_connection:
			self.recurrent_weights = nn.Parameter(
				torch.empty((self.output_size, self.output_size), device=self.device, dtype=torch.float32),
				requires_grad=True
			)
			if use_rec_eye_mask:
				self.rec_mask = (1 - torch.eye(self.output_size, device=self.device, dtype=torch.float32))
			else:
				self.rec_mask = torch.ones(
					(self.output_size, self.output_size), device=self.device,  dtype=torch.float32
				)
		else:
			self.recurrent_weights = None
			self.rec_mask = None

	def _set_default_kwargs(self):
		raise NotImplementedError()

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		raise NotImplementedError

	def _init_forward_state(
			self,
			state: Tuple[torch.Tensor, ...] = None,
			batch_size: int = 1
	) -> Tuple[torch.Tensor, ...]:
		if state is None:
			state = self.create_empty_state(batch_size)
		elif any([e is None for e in state]):
			empty_state = self.create_empty_state(batch_size)
			state = list(state)
			for i, e in enumerate(state):
				if e is None:
					state[i] = empty_state[i]
			state = tuple(state)
		return state

	def forward(self, inputs: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
		raise NotImplementedError

	def initialize_weights_(self):
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param)
			else:
				torch.nn.init.normal_(param)


class LIFLayer(RNNLayer):
	def __init__(
			self,
			input_size: int,
			output_size: int,
			use_recurrent_connection=True,
			use_rec_eye_mask=True,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			dt=1e-3,
			device=None,
			**kwargs
	):
		self.spike_func = spike_func
		super(LIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			dt=dt,
			device=device,
			**kwargs
		)

		self.alpha = torch.tensor(np.exp(-dt / self.kwargs["tau_m"]), dtype=torch.float32, device=self.device)
		self.threshold = torch.tensor(self.kwargs["threshold"], dtype=torch.float32, device=self.device)
		self.gamma = torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device)
		self.initialize_weights_()

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_m", 10.0 * self.dt)
		self.kwargs.setdefault("threshold", 1.0)
		if isinstance(self.spike_func, HeavisideSigmoidApprox):
			self.kwargs.setdefault("gamma", 100.0)
		else:
			self.kwargs.setdefault("gamma", 1.0)

	def initialize_weights_(self):
		gain = self.threshold.data
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param, gain=gain)
			else:
				torch.nn.init.normal_(param, std=gain)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			([membrane potential of shape (batch_size, self.output_size)],
			[spikes of shape (batch_size, self.output_size)])
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		state = tuple([torch.zeros(
			(batch_size, self.output_size),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		) for _ in range(2)])
		return state

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		# state = self._init_forward_state(state, batch_size)
		# next_state = self.create_empty_state(batch_size)
		V, Z = self._init_forward_state(state, batch_size)
		# next_V, next_Z = self.create_empty_state(batch_size)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		# next_V = self.alpha * V + input_current + rec_current - Z.detach() * self.threshold
		next_V = (self.alpha * V + input_current + rec_current) * (1.0 - Z.detach())
		next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)
		return next_Z, (next_V, next_Z)


class ALIFLayer(LIFLayer):
	def __init__(
			self,
			input_size: int,
			output_size: int,
			use_recurrent_connection=True,
			use_rec_eye_mask=True,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			dt=1e-3,
			device=None,
			**kwargs
	):
		super(ALIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			spike_func=spike_func,
			dt=dt,
			device=device,
			**kwargs
		)
		self.beta = torch.tensor(self.kwargs["beta"], dtype=torch.float32, device=self.device)
		if kwargs["learn_beta"]:
			self.beta = torch.nn.Parameter(self.beta, requires_grad=True)
		self.rho = torch.tensor(np.exp(-dt / self.kwargs["tau_a"]), dtype=torch.float32, device=self.device)

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_m", 20.0 * self.dt)
		self.kwargs.setdefault("tau_a", 200.0 * self.dt)
		self.kwargs.setdefault("beta", 1.6)
		self.kwargs.setdefault("threshold", 0.03)
		if isinstance(self.spike_func, HeavisideSigmoidApprox):
			self.kwargs.setdefault("gamma", 100.0)
		else:
			self.kwargs.setdefault("gamma", 0.3)
		self.kwargs.setdefault("learn_beta", False)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[[membrane potential of shape (batch_size, self.output_size)]
			[current threshold of shape (batch_size, self.output_size)]
			[spikes of shape (batch_size, self.output_size)]]
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		state = [torch.zeros(
			(batch_size, self.output_size),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		) for _ in range(3)]
		return tuple(state)

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, a, Z = self._init_forward_state(state, batch_size)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		# v_j^{t+1} = \alpha * v_j^t + \sum_i W_{ji}*z_i^t + \sum_i W_{ji}^{in}x_i^{t+1} - z_j^t * v_{th}
		next_V = (self.alpha * V + input_current + rec_current) * (1.0 - Z.detach())
		next_a = self.rho * a + Z  # a^{t+1} = \rho * a_j^t + z_j^t
		A = self.threshold + self.beta * next_a  # A_j^t = v_{th} + \beta * a_j^t
		next_Z = self.spike_func.apply(next_V, A, self.gamma)  # z_j^t = H(v_j^t - A_j^t)
		return next_Z, (next_V, next_a, next_Z)


class IzhikevichLayer(RNNLayer):
	"""
	Izhikevich p.274

	"""
	def __init__(
			self,
			input_size: int,
			output_size: int,
			use_recurrent_connection=True,
			use_rec_eye_mask=True,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			dt=1e-3,
			device=None,
			**kwargs
	):
		self.spike_func = spike_func
		super(IzhikevichLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=use_recurrent_connection,
			use_rec_eye_mask=use_rec_eye_mask,
			dt=dt,
			device=device,
			**kwargs
		)

		self.C = torch.tensor(self.kwargs["C"], dtype=torch.float32, device=self.device)
		self.v_rest = torch.tensor(self.kwargs["v_rest"], dtype=torch.float32, device=self.device)
		self.v_th = torch.tensor(self.kwargs["v_th"], dtype=torch.float32, device=self.device)
		self.k = torch.tensor(self.kwargs["k"], dtype=torch.float32, device=self.device)
		self.a = torch.tensor(self.kwargs["a"], dtype=torch.float32, device=self.device)
		self.b = torch.tensor(self.kwargs["b"], dtype=torch.float32, device=self.device)
		self.c = torch.tensor(self.kwargs["c"], dtype=torch.float32, device=self.device)
		self.d = torch.tensor(self.kwargs["d"], dtype=torch.float32, device=self.device)
		self.v_peak = torch.tensor(self.kwargs["v_peak"], dtype=torch.float32, device=self.device)
		self.gamma = torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device)
		self.initialize_weights_()

	def _set_default_kwargs(self):
		self.kwargs.setdefault("C", 100.0)
		self.kwargs.setdefault("v_rest", -60.0)
		self.kwargs.setdefault("v_th", -40.0)
		self.kwargs.setdefault("k", 0.7)
		self.kwargs.setdefault("a", 0.03)
		self.kwargs.setdefault("b", -2.0)
		self.kwargs.setdefault("c", -50.0)
		self.kwargs.setdefault("d", 100.0)
		self.kwargs.setdefault("v_peak", 35.0)
		if isinstance(self.spike_func, HeavisideSigmoidApprox):
			self.kwargs.setdefault("gamma", 100.0)
		else:
			self.kwargs.setdefault("gamma", 1.0)

	def initialize_weights_(self):
		gain = 1.0
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param, gain=gain)
			else:
				torch.nn.init.normal_(param, std=gain)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			([membrane potential of shape (batch_size, self.output_size)],
			[membrane potential of shape (batch_size, self.output_size)],
			[spikes of shape (batch_size, self.output_size)])
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		V = self.v_rest * torch.ones(
			(batch_size, self.output_size),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		)
		u = torch.zeros(
			(batch_size, self.output_size),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		)
		Z = torch.zeros(
			(batch_size, self.output_size),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		)
		return V, u, Z

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, u, Z = self._init_forward_state(state, batch_size)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, torch.mul(self.recurrent_weights, self.rec_mask))
		else:
			rec_current = 0.0
		is_reset = Z.detach()
		I = input_current + rec_current
		dVdt = self.k * (V - self.v_rest) * (V - self.v_th) - u + I
		next_V = (V + self.dt * dVdt / self.C) * (1.0 - is_reset) + self.c * is_reset
		dudt = self.a * (self.b * (V - self.v_rest) - u)
		next_u = (u + self.dt * dudt) + self.d * is_reset
		next_Z = self.spike_func.apply(next_V, self.v_peak, self.gamma)
		return next_Z, (next_V, next_u, next_Z)


class LILayer(RNNLayer):
	def __init__(
			self,
			input_size: int,
			output_size: int,
			dt=1e-3,
			device=None,
			**kwargs
	):
		super(LILayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=False,
			dt=dt,
			device=device,
			**kwargs
		)
		self.bias_weights = nn.Parameter(
			torch.empty((self.output_size,), device=self.device),
			requires_grad=True,
		)
		self.kappa = torch.tensor(np.exp(-self.dt / self.kwargs["tau_out"]), dtype=torch.float32, device=self.device)
		self.initialize_weights_()

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_out", 10.0 * self.dt)

	def initialize_weights_(self):
		super(LILayer, self).initialize_weights_()
		torch.nn.init.constant_(self.bias_weights, 0.0)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[membrane potential of shape (batch_size, self.output_size)]
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		state = [torch.zeros(
			(batch_size, self.output_size),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		)]
		return tuple(state)

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		# state = self._init_forward_state(state, batch_size)
		V, = self._init_forward_state(state, batch_size)
		next_V = self.kappa * V + torch.matmul(inputs, self.forward_weights) + self.bias_weights
		return next_V, (next_V, )


LayerType2Layer = {
	LayerType.LIF: LIFLayer,
	LayerType.ALIF: ALIFLayer,
	LayerType.Izhikevich: IzhikevichLayer,
	LayerType.LI: LILayer,
}

