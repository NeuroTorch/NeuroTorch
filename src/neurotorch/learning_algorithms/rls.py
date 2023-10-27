from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable, Tuple

import torch
import torch.nn.functional as F

from .learning_algorithm import LearningAlgorithm
from ..learning_algorithms.tbptt import TBPTT
from ..transforms.base import ToDevice
from ..utils import list_insert_replace_at, ConnectivityConvention, unpack_out_hh, format_pred_batch
from ..utils.autograd import compute_jacobian, filter_parameters, recursive_detach, recursive_detach_


class RLS(TBPTT):
    r"""
    Apply the recursive least squares algorithm to the given model. Different strategies are available to update the
    parameters of the model. The strategy is defined by the :attr:`strategy` attribute of the class. The following
    strategies are available:

        - `inputs`: The parameters are updated using the inputs of the model.
        - `outputs`: The parameters are updated using the outputs of the model. This one is inspired by the work of
            Perich and al. :cite:t:`perich_inferring_2021` with the CURBD algorithm.
        - `grad`: The parameters are updated using the gradients of the model. This one is inspired by the work of
            Zhang and al. :cite:t:`zhang_revisiting_2021`.
        - `jacobian`: The parameters are updated using the Jacobian of the model. This one is inspired by the work of
            Al-Batah and al. :cite:t:`al-batah_modified_2010`.
        - `scaled_jacobian`: The parameters are updated using the scaled Jacobian of the model.

    .. note::
        The `inputs` and `outputs` strategies are limited to an optimization of only one parameter. The others
        strategies can be used with multiple parameters. Unfortunately, those strategies do not work as expected
        at the moment. If you want to help with the development of those strategies, please open an issue on
        GitHub.

    """
    CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
    CHECKPOINT_P_STATES_DICT_KEY: str = "P_list"

    @staticmethod
    def curbd_step_method(
            inv_corr: torch.Tensor,
            post_activation: torch.Tensor,
            error: torch.Tensor,
            connectivity_convention: ConnectivityConvention = ConnectivityConvention.ItoJ,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = torch.mean(post_activation.view(-1, post_activation.shape[-1]), dim=0).view(-1, 1)  # [f_out, 1]
        k = torch.matmul(inv_corr, phi)  # [f_out, f_out] @ [f_out, 1] -> [f_out, 1]
        rPr = torch.matmul(phi.T, k)  # [1, f_out] @ [f_out, 1] -> [1]
        c = 1.0 / (1.0 + rPr)  # [1]
        delta_inv_corr = c * torch.matmul(k, k.T)  # [f_out, 1] @ [1, f_out] -> [f_out, f_out]
        if connectivity_convention == ConnectivityConvention.ItoJ:
            delta_weights = c * torch.outer(k.view(-1), error.view(-1))  # [f_out, 1] @ [1, f_out] -> [N_in, N_out]
        elif connectivity_convention == ConnectivityConvention.JtoI:
            delta_weights = c * torch.outer(error.view(-1), k.view(-1))  # [f_out, 1] @ [1, f_out] -> [N_in, N_out]
        else:
            raise ValueError(f"Invalid connectivity convention: {connectivity_convention}")
        return delta_weights, delta_inv_corr

    def __init__(
            self,
            *,
            params: Optional[Sequence[torch.nn.Parameter]] = None,
            layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
            criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] = None,
            backward_time_steps: Optional[int] = None,
            is_recurrent: bool = False,
            **kwargs
    ):
        """
        Constructor for RLS class.

        :param params: The parameters to optimize. If None, the parameters of the model's trainer will be used.
        :type params: Optional[Sequence[torch.nn.Parameter]]
        :param layers: The layers to optimize. If not None the parameters of the layers will be added to the
            parameters to optimize.
        :type layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]]
        :param criterion: The criterion to use. If not provided, torch.nn.MSELoss is used.
        :type criterion: Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]]
        :param backward_time_steps: The frequency of parameter optimisation. If None, the number of
            time steps of the data will be used.
        :type backward_time_steps: Optional[int]
        :param is_recurrent: If True, the model is recurrent. If False, the model is not recurrent.
        :type is_recurrent: bool
        :param kwargs: The keyword arguments to pass to the BaseCallback.

        :keyword bool save_state: Whether to save the state of the optimizer. Defaults to True.
        :keyword bool load_state: Whether to load the state of the optimizer. Defaults to True.
        """
        kwargs.setdefault("auto_backward_time_steps_ratio", 0)
        kwargs.setdefault("save_state", True)
        kwargs.setdefault("load_state", True)
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(
            params=params,
            layers=layers,
            criterion=criterion,
            backward_time_steps=backward_time_steps,
            optimizer=None,
            optim_time_steps=None,
            **kwargs
        )

        # RLS attributes
        self.P_list = None
        self.delta = kwargs.get("delta", 1.0)
        self.Lambda = kwargs.get("Lambda", 1.0)
        self.kappa = kwargs.get("kappa", 1.0)
        self._device = kwargs.get("device", None)
        self.to_cpu_transform = ToDevice(device=torch.device("cpu"))
        self.to_device_transform = None
        self._other_dims_as_batch = kwargs.get("other_dims_as_batch", False)
        self._is_recurrent = is_recurrent
        self.strategy = kwargs.get("strategy", "inputs").lower()
        self.strategy_to_mth = {
            "inputs": self.inputs_mth_step,
            "outputs": self.outputs_mth_step,
            "grad": self.grad_mth_step,
            "jacobian": self.jacobian_mth_step,
            "scaled_jacobian": self.scaled_jacobian_mth_step,
        }
        self.kwargs = kwargs
        self._asserts()
        self._last_layers_buffer = defaultdict(list)

    def _asserts(self):
        assert 0.0 < self.Lambda <= 1.0, "Lambda must be between 0 and 1"
        assert self.strategy in self.strategy_to_mth, f"Strategy must be one of {list(self.strategy_to_mth.keys())}"

    def __repr__(self):
        repr_str = f"{self.name}: ("
        repr_str += f"priority={self.priority}, "
        repr_str += f"save_state={self.save_state}, "
        repr_str += f"load_state={self.load_state}, "
        repr_str += f"strategy={self.strategy}, "
        repr_str += f"delta={self.delta}, "
        repr_str += f"Lambda={self.Lambda}, "
        repr_str += f"kappa={self.kappa}"
        repr_str += f")"
        return repr_str

    def initialize_P_list(self, m=None):
        self.P_list = [
            self.delta * torch.eye(param.numel() if m is None else m, dtype=torch.float32, device=torch.device("cpu"))
            for param in self.params
        ]

    def _maybe_update_time_steps(self):
        if self._auto_set_backward_time_steps:
            self.backward_time_steps = max(1, int(self._auto_backward_time_steps_ratio * self._data_n_time_steps))

    def _decorate_forward(self, forward, layer_name: str):
        def _forward(*args, **kwargs):
            out = forward(*args, **kwargs)
            t = kwargs.get("t", None)
            if t is None:
                return out
            out_tensor, hh = unpack_out_hh(out)
            list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
            if len(self._layers_buffer[layer_name]) == self.backward_time_steps:
                self._backward_at_t(t, self.backward_time_steps, layer_name)
                if self.strategy in ["grad", "jacobian", "scaled_jacobian"]:
                    out = recursive_detach(out)
            return out
        return _forward

    def _output_hook(self, module, args, kwargs, output) -> None:
        t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
        if t is None:
            return

        layer_name = module.name
        out_tensor, hh = unpack_out_hh(output)
        list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
        if len(self._layers_buffer[layer_name]) == self.backward_time_steps:
            self._backward_at_t(t, self.backward_time_steps, layer_name)
            if self.strategy in ["grad", "jacobian", "scaled_jacobian"]:
                output = recursive_detach_(output)

    def _backward_at_t(self, t: int, backward_t: int, layer_name: str):
        if self._last_layers_buffer[layer_name]:
            x_batch = self._get_x_batch_from_buffer(layer_name)
        else:
            x_batch = self._get_x_batch_slice_from_trainer(0, backward_t, layer_name)
        y_batch = self._get_y_batch_slice_from_trainer((t + 1) - backward_t, t + 1, layer_name)
        pred_batch = self._get_pred_batch_from_buffer(layer_name)
        self.optimization_step(x_batch, pred_batch, y_batch)
        self._last_layers_buffer[layer_name] = self._layers_buffer[layer_name].copy()
        self._layers_buffer[layer_name].clear()

    def _get_x_batch_slice_from_trainer(self, t_first: int, t_last: int, layer_name: str = None):
        x_batch = self.trainer.current_training_state.x_batch
        if isinstance(x_batch, dict):
            if layer_name is None:
                x_batch = {
                    key: val[:, t_first:t_last]
                    for key, val in x_batch.items()
                }
            else:
                x_batch = x_batch[layer_name][:, t_first:t_last]
        else:
            x_batch = x_batch[:, t_first:t_last]
        return x_batch.clone()

    def _get_x_batch_from_buffer(self, layer_name: str):
        pred_batch = torch.stack(self._last_layers_buffer[layer_name], dim=1)
        return pred_batch

    def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
        if self.save_state:
            state = checkpoint.get(self.name, {})
            self.P_list = state.get(self.CHECKPOINT_P_STATES_DICT_KEY, None)

    def get_checkpoint_state(self, trainer, **kwargs) -> object:
        if self.save_state:
            return {
                self.CHECKPOINT_P_STATES_DICT_KEY: self.P_list,
            }
        return None

    def _try_put_on_device(self, trainer):
        try:
            self.P_list = [self.to_device_transform(p) for p in self.P_list]
        except Exception as e:
            trainer.model = self.to_cpu_transform(trainer.model)
            self.P_list = [self.to_device_transform(p) for p in self.P_list]

    def _put_on_cpu(self):
        self.P_list = [self.to_cpu_transform(p) for p in self.P_list]

    def start(self, trainer, **kwargs):
        super().start(trainer, **kwargs)
        if self._device is None:
            self._device = trainer.model.device
        self.to_device_transform = ToDevice(device=self._device)
        self.params = filter_parameters(self.params, requires_grad=True)

    def on_batch_begin(self, trainer, **kwargs):
        LearningAlgorithm.on_batch_begin(self, trainer, **kwargs)
        self.trainer = trainer
        if self._is_recurrent:
            self._data_n_time_steps = self._get_data_time_steps_from_y_batch(
                trainer.current_training_state.y_batch, trainer.current_training_state.x_batch
            )
            self._maybe_update_time_steps()
            self.decorate_forwards()

    def on_batch_end(self, trainer, **kwargs):
        super().on_batch_end(trainer)
        self.undecorate_forwards()
        self._layers_buffer.clear()

    def on_optimization_begin(self, trainer, **kwargs):
        x_batch = trainer.current_training_state.x_batch
        y_batch = trainer.current_training_state.y_batch
        pred_batch = format_pred_batch(trainer.current_training_state.pred_batch, y_batch)

        if self._is_recurrent:
            for layer_name in self._layers_buffer:
                backward_t = len(self._layers_buffer[layer_name])
                if backward_t > 0:
                    self._backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
        else:
            self.optimization_step(x_batch, pred_batch, y_batch)

        trainer.update_state_(batch_loss=self.apply_criterion(pred_batch, y_batch).detach_())

    def optimization_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
        if self.strategy not in self.strategy_to_mth:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        return self.strategy_to_mth[self.strategy](x_batch, pred_batch, y_batch)

    def scaled_jacobian_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        This method is inspired by the work of Al-Batah and al. :cite:t:`al-batah_modified_2010`. Unfortunately, this
        method does not seem to work with the current implementation.

        TODO: Make it work.

        x.shape = [B, f_in]
        y.shape = [B, f_out]
        error.shape = [B, f_out]
        P.shape = [f_out, f_out]
        theta.shape = [ell, 1]

        epsilon = mean[B](error[B, f_out]) -> [1, f_out]
        phi = mean[B](y[B, f_out]) [1, f_out]
        psi = jacobian[theta](phi[1, f_out]]) -> [f_out, ell]

        K = P[f_out, f_out] @ psi[f_out, ell] -> [f_out, ell]
        h = 1 / (labda[1] + kappa[1] * psi.T[ell, f_out] @ K[f_out, ell]) -> [ell, ell]
        grad = (K[f_out, ell] @ h[ell, ell]).T[ell, f_out] @ epsilon.T[f_out, 1] -> [1, ell]
        P = labda[1] * P[f_out, f_out] - kappa[1] * K[f_out, ell] @ h[ell, ell] @ K[f_out, ell].T -> [f_out, f_out]

        In this case f_in must be equal to N_in.

        :param x_batch: inputs of the layer
        :param pred_batch: outputs of the layer
        :param y_batch: targets of the layer

        """
        if getattr(self.trainer, "model", None) is None:
            model_device = pred_batch.device
        else:
            model_device = self.trainer.model.device
        assert isinstance(x_batch, torch.Tensor), "x_batch must be a torch.Tensor"
        assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
        assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
        self.optimizer.zero_grad()

        x_batch_view = x_batch.view(-1, x_batch.shape[-1])  # [B, f_in]
        pred_batch_view = pred_batch.view(-1, pred_batch.shape[-1])  # [B, f_out]
        y_batch_view = y_batch.view(-1, y_batch.shape[-1])  # [B, f_out]
        error = self.to_device_transform(pred_batch_view - y_batch_view)  # [B, f_out]

        if self.P_list is None:
            self.initialize_P_list(m=pred_batch_view.shape[-1])
        self.P_list = self.to_device_transform(self.P_list)
        self.params = self.to_device_transform(self.params)

        epsilon = error.mean(dim=0).view(1, -1)  # [1, f_out]
        phi = pred_batch_view.mean(dim=0).view(1, -1)  # [1, f_out]
        psi_list = compute_jacobian(params=self.params, y=phi.view(-1), strategy="slow")  # [f_out, ell]
        K_list = [torch.matmul(P, psi) for P, psi in zip(self.P_list, psi_list)]  # [f_out, f_out] @ [f_out, ell] -> [f_out, ell]
        h_list = [torch.linalg.pinv(self.Lambda + self.kappa * torch.matmul(psi.T, K)) for psi, K in zip(psi_list, K_list)]  # [ell, f_out] @ [f_out, ell] -> [ell, ell]

        for p, K, h in zip(self.params, K_list, h_list):
            p.grad = torch.matmul(torch.matmul(K, h).T, epsilon.T).view(p.shape).clone()  # ([f_out, ell] @ [ell, ell]).T @ [f_out, 1] -> [ell, 1]

        self.optimizer.step()
        self.P_list = [
            self.Lambda * P - self.kappa * torch.matmul(torch.matmul(K, h), K.T)
            for P, h, K in zip(self.P_list, h_list, K_list)
        ]  # [f_out, f_out] - [f_out, ell] @ [ell, ell] @ [ell, f_out] -> [f_out, f_out]

        self._put_on_cpu()
        if getattr(self.trainer, "model", None) is not None:
            self.trainer.model.to(model_device, non_blocking=True)

    def jacobian_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        This method is inspired by the work of Al-Batah and al. :cite:t:`al-batah_modified_2010`. Unfortunately, this
        method does not seem to work with the current implementation.

        TODO: Make it work.

        x.shape = [B, f_in]
        y.shape = [B, f_out]
        error.shape = [B, f_out]
        P.shape = [f_out, f_out]
        theta.shape = [ell, 1]

        epsilon = mean[B](error[B, f_out]) -> [1, f_out]
        phi = mean[B](y[B, f_out]) [1, f_out]
        psi = jacobian[theta](phi[1, f_out]]) -> [f_out, L]

        K = P[f_out, f_out] @ psi[f_out, L] -> [f_out, L]
        grad = epsilon[1, f_out] @ K[f_out, L] -> [L, 1]
        P = labda[1] * P[f_out, f_out] - kappa[1] * K[f_out, L] @ K[f_out, L].T -> [f_out, f_out]

        In this case f_in must be equal to N_in.

        :param x_batch: inputs of the layer
        :param pred_batch: outputs of the layer
        :param y_batch: targets of the layer

        """
        if getattr(self.trainer, "model", None) is None:
            model_device = pred_batch.device
        else:
            model_device = self.trainer.model.device
        assert isinstance(x_batch, torch.Tensor), "x_batch must be a torch.Tensor"
        assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
        assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
        self.optimizer.zero_grad()

        x_batch_view = x_batch.view(-1, x_batch.shape[-1])  # [B, f_in]
        pred_batch_view = pred_batch.view(-1, pred_batch.shape[-1])  # [B, f_out]
        y_batch_view = y_batch.view(-1, y_batch.shape[-1])  # [B, f_out]
        error = self.to_device_transform(pred_batch_view - y_batch_view)  # [B, f_out]

        if self.P_list is None:
            self.initialize_P_list(m=pred_batch_view.shape[-1])
        self.P_list = self.to_device_transform(self.P_list)
        self.params = self.to_device_transform(self.params)

        epsilon = error.mean(dim=0).view(1, -1)  # [1, f_out]
        phi = pred_batch_view.mean(dim=0).view(1, -1)  # [1, f_out]
        psi_list = compute_jacobian(params=self.params, y=phi.view(-1), strategy="slow")  # [f_out, L]
        K_list = [torch.matmul(P, psi) for P, psi in zip(self.P_list, psi_list)]  # [f_out, f_out] @ [f_out, ell] -> [f_out, L]

        for p, K in zip(self.params, K_list):
            p.grad = torch.matmul(K.T, epsilon.T).view(p.shape).clone()  # [L, f_out] @ [f_out, 1] -> [L, 1]

        self.optimizer.step()
        self.P_list = [
            self.Lambda * P - self.kappa * torch.matmul(K, K.T)
            for P, K in zip(self.P_list, K_list)
        ]  # [f_out, f_out] - [f_out, L] @ [L, f_out] -> [f_out, f_out]

        self._put_on_cpu()
        if getattr(self.trainer, "model", None) is not None:
            self.trainer.model.to(model_device, non_blocking=True)

    def grad_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        This method is inspired by the work of Zhang and al. :cite:t:`zhang_revisiting_2021`. Unfortunately, this
        method does not seem to work with the current implementation.

        TODO: Make it work.

        x.shape = [B, f_in]
        y.shape = [B, f_out]
        error.shape = [B, f_out]
        P.shape = [f_in, f_in]

        epsilon = mean[B](error[B, f_out]) -> [1, f_out]
        phi = mean[B](x[B, f_in]) [1, f_in]

        K = P[f_in, f_in] @ phi.T[f_in, 1] -> [f_in, 1]
        h = 1 / (labda[1] + kappa[1] * phi[1, f_in] @ K[f_in, 1]) -> [1]
        grad = h[1] * P[f_in, f_in] @ grad[N_in, N_out] -> [N_in, N_out]
        P = labda[1] * P[f_in, f_in] - h[1] * kappa[1] * K[f_in, 1] @ K.T[1, f_in] -> [f_in, f_in]

        In this case f_in must be equal to N_in.

        :param x_batch: inputs of the layer
        :param pred_batch: outputs of the layer
        :param y_batch: targets of the layer

        """
        if getattr(self.trainer, "model", None) is None:
            model_device = pred_batch.device
        else:
            model_device = self.trainer.model.device
        assert isinstance(x_batch, torch.Tensor), "x_batch must be a torch.Tensor"
        assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
        assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
        self.optimizer.zero_grad()

        mse_loss = F.mse_loss(pred_batch, y_batch)
        if mse_loss.grad_fn is None:
            # TODO: add a check if it is always the case and if so, warn the user.
            return
        mse_loss.backward()

        x_batch_view = x_batch.view(-1, x_batch.shape[-1])  # [B, f_in]
        pred_batch_view = pred_batch.view(-1, pred_batch.shape[-1])  # [B, f_out]
        y_batch_view = y_batch.view(-1, y_batch.shape[-1])  # [B, f_out]
        error = self.to_device_transform(pred_batch_view - y_batch_view)  # [B, f_out]

        if self.P_list is None:
            self.initialize_P_list(m=x_batch_view.shape[-1])
            for p in self.params:
                # making sur that f_in = N_in.
                if p.shape[0] != x_batch_view.shape[-1]:
                    raise ValueError(
                        f"For inputs of shape [B, f_in], the first dimension of the parameters must be f_in, "
                        f"got {p.shape[0]} instead of {x_batch_view.shape[-1]}."
                    )
        self.P_list = self.to_device_transform(self.P_list)
        self.params = self.to_device_transform(self.params)

        epsilon = error.mean(dim=0).view(1, -1)  # [1, f_out]
        phi = x_batch_view.mean(dim=0).view(1, -1).detach().clone()  # [1, f_in]
        K_list = [torch.matmul(P, phi.T) for P in self.P_list]  # [f_in, f_in] @ [f_in, 1] -> [f_in, 1]
        h_list = [1.0 / (self.Lambda + self.kappa * torch.matmul(phi, K)).item() for K in K_list]  # [1, f_in] @ [f_in, 1] -> [1]

        for p, P, h in zip(self.params, self.P_list, h_list):
            p.grad = h * torch.matmul(P, p.grad)  # [f_in, f_in] @ [N_in, N_out] -> [N_in, N_out]

        self.optimizer.step()
        self.P_list = [
            self.Lambda * P - h * self.kappa * torch.matmul(K, K.T)
            for P, h, K in zip(self.P_list, h_list, K_list)
        ]  # [f_in, 1] @ [1, f_in] -> [f_in, f_in]

        self._put_on_cpu()
        if getattr(self.trainer, "model", None) is not None:
            self.trainer.model.to(model_device, non_blocking=True)

    def inputs_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
        """


        x.shape = [B, f_in]
        y.shape = [B, f_out]
        error.shape = [B, f_out]
        P.shape = [f_in, f_in]

        epsilon = mean[B](error[B, f_out]) -> [1, f_out]
        phi = mean[B](x[B, f_in]) [1, f_in]

        K = P[f_in, f_in] @ phi.T[f_in, 1] -> [f_in, 1]
        h = 1 / (labda[1] + kappa[1] * phi[1, f_in] @ K[f_in, 1]) -> [1]
        P = labda[1] * P[f_in, f_in] - h[1] * kappa[1] * K[f_in, 1] @ K.T[1, f_in] -> [f_in, f_in]
        grad = h[1] * K[f_in, 1] @ epsilon[1, f_out] -> [N_in, N_out]

        In this case [N_in, N_out] must be equal to [f_in, f_out].

        :param x_batch: inputs of the layer
        :param pred_batch: outputs of the layer
        :param y_batch: targets of the layer

        """
        if getattr(self.trainer, "model", None) is None:
            model_device = pred_batch.device
        else:
            model_device = self.trainer.model.device
        assert isinstance(x_batch, torch.Tensor), "x_batch must be a torch.Tensor"
        assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
        assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
        self.optimizer.zero_grad()

        x_batch_view = x_batch.view(-1, x_batch.shape[-1]).detach()  # [B, f_in]
        pred_batch_view = pred_batch.view(-1, pred_batch.shape[-1]).detach()  # [B, f_out]
        y_batch_view = y_batch.view(-1, y_batch.shape[-1]).detach()  # [B, f_out]
        error = self.to_device_transform(pred_batch_view - y_batch_view)  # [B, f_out]

        if self.P_list is None:
            self.initialize_P_list(m=x_batch_view.shape[-1])
            for p in self.params:
                # making sur that f_in = N_in.
                if p.shape[0] != x_batch_view.shape[-1]:
                    raise ValueError(
                        f"For inputs of shape [B, f_in], the first dimension of the parameters must be f_in, "
                        f"got {p.shape[0]} instead of {x_batch_view.shape[-1]}."
                    )
                # making sure that f_out = N_out.
                if p.shape[1] != y_batch_view.shape[-1]:
                    raise ValueError(
                        f"For targets of shape [B, f_out], the second dimension of the parameters must be f_out, "
                        f"got {p.shape[1]} instead of {y_batch_view.shape[-1]}."
                    )
        self.P_list = self.to_device_transform(self.P_list)
        self.params = self.to_device_transform(self.params)

        epsilon = error.mean(dim=0).view(1, -1)  # [1, f_out]
        phi = x_batch_view.mean(dim=0).view(1, -1).detach().clone()  # [1, f_in]
        K_list = [torch.matmul(P, phi.T) for P in self.P_list]  # [f_in, f_in] @ [f_in, 1] -> [f_in, 1]
        h_list = [1.0 / (self.Lambda + self.kappa * torch.matmul(phi, K)).item() for K in K_list]  # [1, f_in] @ [f_in, 1] -> [1]

        for p, K, h in zip(self.params, K_list, h_list):
            p.grad = h * torch.outer(K.view(-1), epsilon.view(-1))  # [f_in, 1] @ [1, f_out] -> [N_in, N_out]

        self.optimizer.step()
        self.P_list = [
            self.Lambda * P - h * self.kappa * torch.matmul(K, K.T)
            for P, h, K in zip(self.P_list, h_list, K_list)
        ]  # [f_in, 1] @ [1, f_in] -> [f_in, f_in]

        self._put_on_cpu()
        if getattr(self.trainer, "model", None) is not None:
            self.trainer.model.to(model_device, non_blocking=True)

    def outputs_mth_step(self, x_batch: torch.Tensor, pred_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        This method is inspired by the work of Perich and al. :cite:t:`perich_inferring_2021` with
        the CURBD algorithm.

        x.shape = [B, f_in]
        y.shape = [B, f_out]
        error.shape = [B, f_out]
        epsilon = mean[B](error[B, f_out]) -> [1, f_out]
        phi = mean[B](y[B, f_out]) [1, f_out]

        P.shape = [f_out, f_out]
        K = P[f_out, f_out] @ phi.T[f_out, 1] -> [f_out, 1]
        h = 1 / (labda[1] + kappa[1] * phi[1, f_out] @ K[f_out, 1]) -> [1]
        P = labda[1] * P[f_out, f_out] - h[1] * kappa[1] * K[f_out, 1] @ K.T[1, f_out] -> [f_out, f_out]
        grad = h[1] * K[f_out, 1] @ epsilon[1, f_out] -> [N_in, N_out]

        In this case [N_in, N_out] must be equal to [f_out, f_out].

        :param x_batch: inputs of the layer
        :param pred_batch: outputs of the layer
        :param y_batch: targets of the layer

        """
        if getattr(self.trainer, "model", None) is None:
            model_device = pred_batch.device
        else:
            model_device = self.trainer.model.device
        assert isinstance(x_batch, torch.Tensor), "x_batch must be a torch.Tensor"
        assert isinstance(pred_batch, torch.Tensor), "pred_batch must be a torch.Tensor"
        assert isinstance(y_batch, torch.Tensor), "y_batch must be a torch.Tensor"
        self.optimizer.zero_grad()

        x_batch_view = x_batch.view(-1, x_batch.shape[-1]).detach()  # [B, f_in]
        pred_batch_view = pred_batch.view(-1, pred_batch.shape[-1]).detach()  # [B, f_out]
        y_batch_view = y_batch.view(-1, y_batch.shape[-1]).detach()  # [B, f_out]
        error = self.to_device_transform(pred_batch_view - y_batch_view)  # [B, f_out]

        if self.P_list is None:
            self.initialize_P_list(m=pred_batch_view.shape[-1])
            for p in self.params:
                # making sur that f_out = N_in.
                if p.shape[0] != pred_batch_view.shape[-1]:
                    raise ValueError(
                        f"For inputs of shape [B, f_in], the first dimension of the parameters must be f_in, "
                        f"got {p.shape[0]} instead of {x_batch_view.shape[-1]}."
                    )
                # making sure that f_out = N_out.
                if p.shape[1] != pred_batch_view.shape[-1]:
                    raise ValueError(
                        f"For targets of shape [B, f_out], the second dimension of the parameters must be f_out, "
                        f"got {p.shape[1]} instead of {y_batch_view.shape[-1]}."
                    )
        assert len(self.params) == 1, "This method support the optimization of only one recurrent matrix."
        assert len(self.P_list) == len(self.params), "The number of parameters and P matrices must be equal."
        self.P_list = self.to_device_transform(self.P_list)
        self.params = self.to_device_transform(self.params)

        epsilon = error.mean(dim=0).view(1, -1)  # [1, f_out]

        cubd_objs = [
            self.curbd_step_method(P, pred_batch_view, epsilon)
            for P in self.P_list
        ]
        for i, (p, (delta_weights, delta_inv_corr)) in enumerate(zip(self.params, cubd_objs)):
            p.grad = delta_weights
            self.P_list[i] = self.Lambda * self.P_list[i] - self.kappa * delta_inv_corr

        # phi = pred_batch_view.mean(dim=0).view(1, -1).detach().clone()  # [1, f_out]
        # K_list = [torch.matmul(P, phi.T) for P in self.P_list]  # [f_out, f_out] @ [f_out, 1] -> [f_out, 1]
        # h_list = [1.0 / (self.Lambda + self.kappa * torch.matmul(phi, K)).item() for K in K_list]  # [1, f_out] @ [f_out, 1] -> [1]
        #
        # for p, K, h in zip(self.params, K_list, h_list):
        # 	p.grad = h * torch.outer(K.view(-1), epsilon.view(-1))  # [f_out, 1] @ [1, f_out] -> [N_in, N_out]

        self.optimizer.step()
        # self.P_list = [
        # 	self.Lambda * P - h * self.kappa * torch.matmul(K, K.T)
        # 	for P, h, K in zip(self.P_list, h_list, K_list)
        # ]  # [f_in, 1] @ [1, f_in] -> [f_in, f_in]

        self._put_on_cpu()
        if getattr(self.trainer, "model", None) is not None:
            self.trainer.model.to(model_device, non_blocking=True)

