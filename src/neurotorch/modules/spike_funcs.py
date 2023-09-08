import enum
from typing import Any

import torch


class SpikeFuncType(enum.Enum):
    FastSigmoid = 0
    Phi = 1


class SpikeFunction(torch.autograd.Function):
    """
    Implementation of the spike function. The spike function is a differentiable version of the Heaviside function.
    The Heaviside function is defined as the equation :eq:`heaviside`. The backward pass of this function has to be
    an approximation of the derivative of the Heaviside function.

    .. math::
        :label: heaviside

        \\begin{equation}
            H(x, thr) = \\left\\{
            \\begin{matrix}
                1 & \\text{ if } x > thr; \\\\
                0 & \\text{ else}.
            \\end{matrix}
            \\right.
        \\end{equation}

    """

    @staticmethod
    def forward(
            ctx: torch.autograd.function.FunctionCtx,
            inputs: torch.Tensor,
            threshold: torch.Tensor = torch.tensor(1.0),
            gamma: torch.Tensor = torch.tensor(1.0)
    ) -> torch.Tensor:
        """
        The forward pass of the spike function is the Heaviside function. See the heaviside equation.

        :param ctx: The context of the function. It is used to store information for the backward pass. Use the method
            :func:`ctx.save_for_backward` to store information.
        :type ctx: torch.autograd.function.FunctionCtx
        :param inputs: The input tensor.
        :type inputs: torch.Tensor
        :param threshold: The threshold of the spike function.
        :type threshold: torch.Tensor
        :param gamma: The gamma parameter of the spike function. This parameter is used in the backward pass to
            increase the gradient of the spike function. See child classes for more information.
        :type gamma: torch.Tensor

        :return: The output of the spike function.
        :rtype: torch.Tensor
        """
        ctx.save_for_backward(inputs, threshold, gamma)
        out = torch.zeros_like(inputs)
        out[inputs >= threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_outputs):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        raise NotImplementedError()


# @staticmethod
# def symbolic(g, inputs: torch._C.Value) -> torch._C.Value:
# 	return g.op("SpikeFunction", inputs, g.op("Constant", value_t=torch.tensor(0, dtype=torch.float)))


class HeavisideSigmoidApprox(SpikeFunction):
    """
    Implementation of the spike function. The spike function is a differentiable version of the Heaviside function.
    The Heaviside function is defined in the doc of :class:`SpikeFunction`. The backward pass of this function is the
    first derivative of the fast sigmoid function defined in equation :eq:`fast_sigmoid`. The derivative
    is shown in equation :eq:`fast_sigmoid_derivative` used in Zenke & Ganguli (2018).

    .. math::
        \\begin{equation}
            S(x) = \\frac{1}{1 + e^{-x}}
        \\end{equation}
        :label: fast_sigmoid

    .. math::
        \\begin{equation}
            S'(x) \\approx \\frac{x}{\\left(1 + \\gamma\\vert{x - thr}\\vert\\right)^2}
        \\end{equation}
        :label: fast_sigmoid_derivative

    """

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_outputs: torch.Tensor) -> Any:
        """
        The implementation of the equation :eq:`fast_sigmoid_derivative`.

        :param ctx: The context of the function. It is used to retrieve information from the forward pass.
        :type ctx: torch.autograd.function.FunctionCtx
        :param grad_outputs: The gradient of the loss with respect to the output of the forward pass.
        :type grad_outputs: torch.Tensor
        :return: The gradient of the loss with respect to the input of the forward pass.
        """
        inputs, threshold, scale = ctx.saved_tensors
        grad_outputs_clone = grad_outputs.clone()
        inputs_grad = grad_outputs_clone / (scale * torch.abs(inputs - threshold) + 1.0) ** 2
        if ctx.needs_input_grad[1]:
            threshold_grad = -inputs_grad.clone()
        else:
            threshold_grad = None
        return inputs_grad, threshold_grad, None


class HeavisidePhiApprox(SpikeFunction):
    """
    Implementation of the spike function. The spike function is a differentiable version of the Heaviside function.
    The Heaviside function is defined in the doc of :class:`SpikeFunction`. The backward pass of this function is the
    approximation of the heaviside used in :cite:t:`bellec_solution_2020`. This approximation is defined in equation
    :eq:`heaviside_phi_approx`.

    .. math::
        \\begin{equation}
            \\psi_j^t = \\frac{\\gamma_\\text{pd}}{v_{\\text{th}}}
            \\text{max}\\left(0, 1 - \\left\\vert\\frac{v_j^t - A_j^t}{v_\\text{th}}\\right\\vert\\right)
        \\end{equation}
        :label: heaviside_phi_approx

    .. bibliography::

    """
    epsilon = 1e-5

    @staticmethod
    def pseudo_derivative(inputs, threshold, gamma):
        return (gamma / (threshold + HeavisidePhiApprox.epsilon)) * torch.max(
            torch.zeros_like(inputs), 1 - torch.abs((inputs - threshold) / (threshold + HeavisidePhiApprox.epsilon))
        )

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_outputs):
        """
        The implementation of the equation :eq:`heaviside_phi_approx`.

        :param ctx: The context of the function. It is used to retrieve information from the forward pass.
        :type ctx: torch.autograd.function.FunctionCtx
        :param grad_outputs: The gradient of the loss with respect to the output of the forward pass.
        :type grad_outputs: torch.Tensor
        :return: The gradient of the loss with respect to the input of the forward pass.
        """
        inputs, threshold, gamma = ctx.saved_tensors
        inputs_grad = grad_outputs.clone() * (gamma / (threshold + HeavisidePhiApprox.epsilon)) * torch.max(
            torch.zeros_like(inputs), 1 - torch.abs((inputs - threshold) / (threshold + HeavisidePhiApprox.epsilon))
        )
        if ctx.needs_input_grad[1]:
            threshold_grad = -inputs_grad.clone()
        else:
            threshold_grad = None
        return inputs_grad, threshold_grad, None


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
    X = torch.tensor(np.linspace(th - 2, th + 2, num=1_000), requires_grad=False)
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
