import torch
import torch.nn as nn
from typing import Optional, Union, Iterable, Dict, List
import pythonbasictools as pybt
from . import BaseRegularization
from ..transforms.base import to_numpy
from ..init import dale_


# @pybt.docstring.inherit_fields_docstring(fields=["Attributes"], bases=[BaseRegularization])
class DaleLawL2(BaseRegularization):
    """
    Regularisation of the connectome to apply Dale's law and L2. In a nutshell, the Dale's law
    stipulate that neurons can either have excitatory or inhibitory connections, not both.
    The L2 regularisation reduce the energy of the network. This regularisation allow you to follow the
    Dale's law and/or L2 depending on the factor alpha. The equation used is showed by :eq:`dale_l2`.

    .. math::
        :label: dale_l2

        \\begin{equation}
            \\mathcal{L}_{\\text{DaleLawL2}} = \\text{Tr}\\left( W^T \\left(\\alpha W - \\left(1 - \\alpha\\right) W_{\\text{ref}}\\right) \\right)
        \\end{equation}


    In the case where :math:`\\alpha = 0`, the regularisation will only follow the Dale's law shown by :eq:`dale`.

    .. math::
        :label: dale

        \\begin{equation}
            \\mathcal{L}_{\\text{DaleLaw}} = -\\text{Tr}\\left( W^T W_{\\text{ref}}\\right)
        \\end{equation}

    In the case where :math:`\\alpha = 1`, the regularisation will only follow the L2 regularisation shown by :eq:`l2`.

    .. math::
        :label: l2

        \\begin{equation}
            \\mathcal{L}_{\\text{L2}} = \\text{Tr}\\left( W^T W\\right)
        \\end{equation}


    :Attributes:
        - :attr:`alpha` (float): Number between 0 and 1 that favors one of the constraints.
        - :attr:`dale_kwargs` (dict): kwargs of the Dale's law. See :func:`dale_`.
        - :attr:`reference_weights` (Iterable[torch.Tensor]): Reference weights to compare. Must be the same size as the weights.

    """
    def __init__(
            self,
            params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
            alpha: float = 0.8,
            reference_weights: Optional[Iterable[torch.Tensor]] = None,
            Lambda: float = 1.0,
            optimizer: Optional[torch.optim.Optimizer] = None,
            **dale_kwargs
    ):
        """
        :param params: Weights matrix to regularize (can be multiple)
        :type params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]]
        :param alpha: Number between 0 and 1 that favors one of the constraints.
            If alpha = 0 -> Only Dale's law is applied.
            If alpha = 1 -> Only the reduction of the energy is applied.
            If 1 < alpha < 0 -> Both Dale's law and the reduction of the energy are applied with their ratio.
        :type alpha: float
        :param reference_weights: Reference weights to compare. Must be the same size as the weights. If not provided,
            the weights will be generated automatically with the dale_kwargs.
        :type reference_weights: Optional[Iterable[torch.Tensor]]
        :param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
        :type Lambda: float
        :param dale_kwargs: kwargs of the Dale's law.

        :keyword float inh_ratio: ratio of inhibitory connections. Must be between 0 and 1.
        :keyword float rho: The connectivity ratio. Must be between 0 and 1. If rho = 1, the tensor will be fully connected.
        :keyword bool inh_first: If True, the inhibitory neurons will be in the first half of the tensor. If False,
            the neurons will be shuffled.
        :keyword Optional[int] seed: seed for the random number generator. If None, the seed is not set.
        """
        super(DaleLawL2, self).__init__(params, Lambda, optimizer=optimizer)
        self.__name__ = self.__class__.__name__
        self.alpha = alpha
        if self.alpha > 1 or self.alpha < 0:
            raise ValueError("alpha must be between 0 and 1")
        self.dale_kwargs = dale_kwargs
        self.reference_weights = self._init_reference_weights(reference_weights)

    def _init_reference_weights(
            self,
            reference_weights: Optional[Union[Iterable[torch.Tensor], Dict[str, torch.Tensor]]] = None
    ):
        """
        Initialize the reference weights with Dale's law.
        """
        if reference_weights is None:
            self.reference_weights = []
            for param in self.params:
                self.reference_weights.append(torch.sign(dale_(torch.empty_like(param), **self.dale_kwargs)))
        else:
            self.reference_weights = [torch.sign(ref) for ref in reference_weights]
        return self.reference_weights

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the forward pass of the Dale's law. If alpha = 1 and the reference weights is not provided, it will be
        modified to 0, so it can get cancel.

        :param args: weights matrix
        :param kwargs: kwargs of the forward pass
        """
        loss_list = []
        for param, ref in zip(self.params, self.reference_weights):
            loss = torch.trace(
                param.T @ (self.alpha * param - (1 - self.alpha) * ref.to(param.device))
            )
            loss_list.append(loss)
        if len(self.params) == 0:
            loss = torch.tensor(0.0, dtype=torch.float32)
        else:
            loss = torch.sum(torch.stack(loss_list))
        return loss


class DaleLaw(DaleLawL2):
    def __init__(
            self,
            params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
            reference_weights: Optional[Iterable[torch.Tensor]] = None,
            Lambda: float = 1.0,
            **dale_kwargs
    ):
        """
        :param params: Weights matrix to regularize (can be multiple)
        :param reference_weights: Reference weights to compare. Must be the same size as the weights. If not provided,
            the weights will be generated automatically with the dale_kwargs.
        :param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
        :param dale_kwargs: kwargs of the Dale's law.

        :keyword inh_ratio: ratio of inhibitory connections. Must be between 0 and 1.
        :keyword rho: The connectivity ratio. Must be between 0 and 1. If rho = 1, the tensor will be fully connected.
        :keyword inh_first: If True, the inhibitory neurons will be in the first half of the tensor. If False,
            the neurons will be shuffled.
        :keyword seed: seed for the random number generator. If None, the seed is not set.
        """
        super(DaleLaw, self).__init__(params, 0.0, reference_weights, Lambda, **dale_kwargs)
        self.__name__ = self.__class__.__name__


class WeightsDistance(BaseRegularization):
    def __init__(
            self,
            params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
            reference_weights: Iterable[torch.Tensor],
            Lambda: float = 1.0,
            p: int = 1,
    ):
        super(WeightsDistance, self).__init__(params, Lambda)
        self.reference_weights = list(reference_weights)
        self.p = p

    def forward(self, *args, **kwargs) -> torch.Tensor:
        loss_list = []
        for param, ref in zip(self.params, self.reference_weights):
            loss_list.append(torch.linalg.norm(torch.abs(param - ref), self.p))
        if len(self.params) == 0:
            loss = torch.tensor(0.0, dtype=torch.float32)
        else:
            loss = torch.sum(torch.stack(loss_list))
        return loss


class ExcRatioTargetRegularization(BaseRegularization):
    r"""
    Applies the function:

    .. math::
        \text{loss}(x) = \lambda \cdot \sum_{i=1}^N \left|(\text{mean}(\text{sign}(x_i)) + 1) - 2\cdot\text{target} \right|

    Where :math:`x` is the list of input parameters, :math:`N` is the number of parameter, :math:`\text{sign}(x_i)` is the
    sign of the element :math:`x_i`, :math:`\text{mean}(\text{sign}(x_i))` is the mean of the signs of the elements in the
    tensor, :math:`\text{target}` is the target value and :math:`\lambda` is the weight of the regularization.


    Examples::
        >>> import neurotorch as nt
        >>> layer = nt.WilsonCowanLayer(10, 10, force_dale_law=True)
        >>> m = ExcRatioTargetRegularization(params=layer.get_sign_parameters(), Lambda=0.1, exc_target_ratio=0.9)
        >>> loss = m()
    """
    def __init__(
            self,
            params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
            exc_target_ratio: float = 0.8,
            Lambda: float = 1.0,
            **kwargs
    ):
        """
        Create a new ExcRatioTargetRegularization.

        :param params: Weights matrix to regularize.
        :type params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]]
        :param exc_target_ratio: Target ratio of excitatory neurons. Must be between 0 and 1.
        :type exc_target_ratio: float
        :param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
        :type Lambda: float

        :keyword kwargs: kwargs of the BaseRegularization.
        """
        super(ExcRatioTargetRegularization, self).__init__(params, Lambda, **kwargs)
        assert 0 < exc_target_ratio < 1, "exec_target_ratio must be between 0 and 1"
        self.exc_target_ratio = exc_target_ratio
        self.sign_func = kwargs.get("sign_func", torch.nn.Softsign())

    def forward(self, *args, **kwargs) -> torch.Tensor:
        loss_list = []
        for param in self.params:
            param_ratio = torch.mean(self.sign_func(param)) + 1
            loss_list.append(torch.abs(param_ratio - 2 * self.exc_target_ratio))
        if len(self.params) == 0:
            loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        else:
            loss = torch.sum(torch.stack(loss_list))
        return loss

    def get_params_exc_ratio(self) -> List[float]:
        """
        Returns the excitatory ratio of each parameter.
        """
        return [to_numpy(((torch.mean(self.sign_func(param)) + 1)/2).item()) for param in self.params]

    def get_params_inh_ratio(self) -> List[float]:
        """
        Returns the inhibitory ratio of each parameter.
        """
        return [to_numpy(((1 - torch.mean(self.sign_func(param)))/2).item()) for param in self.params]

    def on_pbar_update(self, trainer, **kwargs) -> dict:
        loss = to_numpy(self().item())
        exc_ratio = self.get_params_exc_ratio()
        return {"exc_ratio": exc_ratio, "exc_ratio_loss": loss}


class InhRatioTargetRegularization(ExcRatioTargetRegularization):
    """
    Applies the `ExcRatioTargetRegularization` with the target ratio of inhibitory neurons.
    """
    def __init__(
            self,
            params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]],
            inh_target_ratio: float = 0.2,
            Lambda: float = 1.0,
    ):
        """
        Create a new InhRatioTargetRegularization.

        :param params: Weights matrix to regularize.
        :type params: Union[Iterable[torch.nn.Parameter], Dict[str, torch.nn.Parameter]]
        :param inh_target_ratio: Target ratio of inhibitory neurons. Must be between 0 and 1.
        :type inh_target_ratio: float
        :param Lambda: The weight of the regularization. In other words, the coefficient that multiplies the loss.
        :type Lambda: float
        """
        assert 0 < inh_target_ratio < 1, "inh_target_ratio must be between 0 and 1"
        super(InhRatioTargetRegularization, self).__init__(
            params=params,
            Lambda=Lambda,
            exc_target_ratio=1 - inh_target_ratio,
        )

    def on_pbar_update(self, trainer, **kwargs) -> dict:
        loss = to_numpy(self().item())
        inh_ratio = self.get_params_inh_ratio()
        return {"inh_ratio": inh_ratio, "inh_ratio_loss": loss}


