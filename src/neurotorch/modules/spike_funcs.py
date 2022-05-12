from typing import Any

import torch
import enum


class SpikeFuncType(enum.Enum):
	FastSigmoid = 0
	Phi = 1


class SpikeFunction(torch.autograd.Function):
	@staticmethod
	def forward(
			ctx: Any,
			inputs: torch.Tensor,
			threshold: torch.Tensor = torch.tensor(1.0),
			gamma: torch.Tensor = torch.tensor(0.3)
	):
		"""
		In the forward pass we compute a step function of the input Tensor
		and return it. ctx is a context object that we use to stash information which
		we need to later backpropagate our error signals. To achieve this we use the
		ctx.save_for_backward method.
		"""
		ctx.save_for_backward(inputs, threshold, gamma)
		out = torch.zeros_like(inputs)
		out[inputs >= threshold] = 1.0
		return out

	@staticmethod
	def backward(ctx: Any, grad_outputs):
		"""
		In the backward pass we receive a Tensor we need to compute the
		surrogate gradient of the loss with respect to the input.
		Here we use the normalized negative part of a fast sigmoid
		as this was done in Zenke & Ganguli (2018).
		"""
		raise NotImplementedError

	# @staticmethod
	# def symbolic(g, inputs: torch._C.Value) -> torch._C.Value:
	# 	return g.op("SpikeFunction", inputs, g.op("Constant", value_t=torch.tensor(0, dtype=torch.float)))


class HeavisideSigmoidApprox(SpikeFunction):
	@staticmethod
	def backward(ctx: Any, grad_outputs):
		"""
		In the backward pass we receive a Tensor we need to compute the
		surrogate gradient of the loss with respect to the input.
		Here we use the normalized negative part of a fast sigmoid
		as this was done in Zenke & Ganguli (2018).

		S(x) = 1 / (1 + e^{-x})
		f(x) \approx S'(x)
		f(x) = x / (1 + abs(x))
		"""
		inputs, threshold, scale = ctx.saved_tensors
		grad_inputs = grad_outputs.clone()
		grad = grad_inputs / (scale * torch.abs(inputs - threshold) + 1.0) ** 2
		return grad, None, None


class HeavisidePhiApprox(SpikeFunction):
	epsilon = 1e-5

	@staticmethod
	def backward(ctx: Any, grad_outputs):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		inputs, threshold, gamma = ctx.saved_tensors
		grad = grad_outputs.clone() * (gamma/(threshold + HeavisidePhiApprox.epsilon)) * torch.max(
			torch.zeros_like(inputs),  1 - torch.abs((inputs - threshold) / (threshold + HeavisidePhiApprox.epsilon))
		)
		return grad, None, None


SpikeFuncType2Func = {
	SpikeFuncType.FastSigmoid: HeavisideSigmoidApprox,
	SpikeFuncType.Phi: HeavisidePhiApprox,
}


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import numpy as np

	th, sc = torch.tensor(1.0), torch.tensor(1.0)
	funcs = {
		"FastSigmoid": HeavisideSigmoidApprox.apply,
		"Bellec": HeavisidePhiApprox.apply
	}
	grads = {
		name: [] for name in funcs
	}
	X = torch.tensor(np.linspace(th-2, th+2, num=1_000), requires_grad=False)
	Y = SpikeFunction.apply(X, th, sc)
	for name, func in funcs.items():
		for x_i in X:
			x = torch.tensor(x_i.clone().detach(), requires_grad=True)
			y = func(x, th, sc)
			# y.retain_grad()
			y.backward()
			grads[name].append(x.grad.detach().cpu().numpy())

	plt.plot(X, Y, label="Heaviside")
	for name in funcs:
		plt.plot(X, grads[name], label=name)
	plt.xlabel("V [mV]")
	plt.legend()
	plt.show()




