from typing import List, Iterable, Optional, Sequence, Union, Tuple

import torch


def zero_grad_params(params: Iterable[torch.nn.Parameter]):
    """
    Set the gradient of the parameters to zero.

    :param params: The parameters to set the gradient to zero.
    """
    for p in params:
        p.grad = torch.zeros_like(p).detach()


def compute_jacobian(
        *,
        model: Optional[torch.nn.Module] = None,
        params: Optional[Iterable[torch.nn.Parameter]] = None,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        strategy: str = "slow",
):
    """
    Compute the jacobian of the model with respect to the parameters.

    # TODO: check https://medium.com/@monadsblog/pytorch-backward-function-e5e2b7e60140
    # TODO: see https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py

    :param model: The model to compute the jacobian.
    :param params: The parameters to compute the jacobian with respect to. If None, compute the jacobian
        with respect to all the parameters of the model.
    :param x: The input to compute the jacobian. If None, use y instead.
    :param y: The output to compute the jacobian. If None, use x instead.
    :param strategy: The strategy to use to compute the jacobian. Can be "slow" or "fast". At this time the only
        strategy implemented is "slow".

    :return: The jacobian.
    """
    if params is None:
        assert model is not None, "If params is None, model must be provided."
        params = model.parameters()
    zero_grad_params(params)

    if y is not None:
        if strategy.lower() == "fast":
            y.backward(torch.ones_like(y))
            jacobian = [p.grad.view(-1) for p in params]
        elif strategy.lower() == "slow":
            jacobian = [[] for _ in range(len(list(params)))]
            grad_outputs = torch.eye(y.shape[-1], device=y.device)
            for i in range(y.shape[-1]):
                zero_grad_params(params)
                y.backward(grad_outputs[i], retain_graph=True)
                for p_idx, param in enumerate(params):
                    jacobian[p_idx].append(param.grad.view(-1).detach().clone())
            jacobian = [torch.stack(jacobian[i], dim=-1).T for i in range(len(list(params)))]
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
    elif x is not None:
        jacobian = torch.autograd.functional.jacobian(model, x, params)
    else:
        raise ValueError("Either x or y must be provided.")
    return jacobian


def dy_dw_local(
        y: torch.Tensor,
        params: Sequence[torch.nn.Parameter],
        grad_outputs: Optional[torch.Tensor] = None,
        retain_graph: bool = True,
        allow_unused: bool = True,
) -> List[torch.Tensor]:
    """
    Compute the derivative of z with respect to the parameters using torch.autograd.grad. If a parameter not
    requires grad, the derivative is set to zero.

    :param y: The tensor to compute the derivative.
    :type y: torch.Tensor
    :param params: The parameters to compute the derivative with respect to.
    :type params: Sequence[torch.nn.Parameter]
    :param grad_outputs: The gradient of the output. If None, use a tensor of ones.
    :type grad_outputs: torch.Tensor or None
    :param retain_graph: If True, the graph used to compute the grad will be retained.
    :type retain_graph: bool
    :param allow_unused: If True, allow the computation of the derivative with respect to a parameter that is not
        used in the computation of z.
    :type allow_unused: bool
    :return: The derivative of z with respect to the parameters.
    :rtype: List[torch.Tensor]
    """
    grad_outputs = torch.ones_like(y) if grad_outputs is None else grad_outputs
    grads_local = []
    for param_idx, param in enumerate(params):
        grad = None
        if param.requires_grad:
            grad = torch.autograd.grad(
                y, param,
                grad_outputs=grad_outputs,
                retain_graph=retain_graph,
                allow_unused=allow_unused,
            )[0]
        if grad is None:
            grad = torch.zeros_like(param)
        grads_local.append(grad)
    return grads_local


def vmap(f):
    # TODO: replace by torch.vmap when it is available
    def wrapper(batch_tensor):
        return torch.stack([f(batch_tensor[i]) for i in range(batch_tensor.shape[0])])

    return wrapper


def filter_parameters(
        parameters: Union[Sequence[torch.nn.Parameter], torch.nn.ParameterList],
        requires_grad: bool = True
) -> List[torch.nn.Parameter]:
    """
    Filter the parameters by their requires_grad attribute.

    :param parameters: The parameters to filter.
    :param requires_grad: The value of the requires_grad attribute to filter.

    :return: The filtered parameters.
    """
    return [p for p in parameters if p.requires_grad == requires_grad]


def get_contributing_params(y, top_level=True):
    """
    Get the parameters that contribute to the computation of y.

    Taken from "https://stackoverflow.com/questions/72301628/find-pytorch-model-parameters-that-dont-contribute-to-loss".

    :param y: The tensor to compute the contribution of the parameters.
    :param top_level: Whether y is a top level tensor or not.
    :type top_level: bool
    :return: A generator of the parameters that contribute to the computation of y.
    """
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)


def recursive_detach(tensors: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]):
    if isinstance(tensors, tuple):
        out = tuple([recursive_detach(o) for o in tensors])
    elif isinstance(tensors, list):
        out = [recursive_detach(o) for o in tensors]
    elif isinstance(tensors, dict):
        out = {k: recursive_detach(v) for k, v in tensors.items()}
    elif isinstance(tensors, torch.Tensor):
        out = tensors.detach()
    else:
        out = tensors
    return out


def recursive_detach_(tensors: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]):
    if isinstance(tensors, tuple):
        out = tuple([recursive_detach_(o) for o in tensors])
    elif isinstance(tensors, list):
        out = [recursive_detach_(o) for o in tensors]
    elif isinstance(tensors, dict):
        out = {k: recursive_detach_(v) for k, v in tensors.items()}
    elif isinstance(tensors, torch.Tensor):
        out = tensors.detach_()
    else:
        out = tensors
    return out
