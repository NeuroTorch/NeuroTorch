import warnings
from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Callable, List, Mapping, Any

import torch

from .learning_algorithm import LearningAlgorithm
from ..transforms.base import to_numpy, to_tensor
from ..learning_algorithms.tbptt import TBPTT
from ..utils import (
    list_insert_replace_at,
    zero_grad_params,
    unpack_out_hh,
    recursive_detach,
    dy_dw_local,
    clip_tensors_norm_,
    recursive_detach_
)
from ..utils.random import unitary_rn_normal_matrix


class Eprop(TBPTT):
    r"""
    Apply the eligibility trace forward propagation (e-prop) :cite:t:`bellec_solution_2020`
    algorithm to the given model.


    .. image:: ../../images/learning_algorithms/EpropDiagram.png
        :width: 300
        :align: center


    Note: If this learning algorithm is used for classification, the output layer should have a log-softmax activation
        function and the target should be a one-hot encoded tensor. Then, the loss function should be the negative log
        likelihood loss function from :class:`nt.losses.NLLLoss` with the ``target_as_one_hot`` argument set to ``True``.

    """
    CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: str = "optimizer_state_dict"
    CHECKPOINT_FEEDBACK_WEIGHTS_KEY: str = "feedback_weights"
    OPTIMIZER_PARAMS_GROUP_IDX = 0
    OPTIMIZER_OUTPUT_PARAMS_GROUP_IDX = 1
    DEFAULT_OPTIMIZER_CLS = torch.optim.AdamW
    DEFAULT_Y_KEY = "default_key"
    DEFAULT_FEEDBACKS_GEN_STRATEGY = "xavier_normal"
    FEEDBACKS_GEN_FUNCS = {
        "randn": lambda *args, **kwargs: torch.randn(*args, **kwargs),
        "xavier_normal": lambda *args, **kwargs: torch.nn.init.xavier_normal_(
            torch.empty(*args), gain=kwargs.get("gain", 1.0)
        ),
        "kaiming_normal": lambda *args, **kwargs: torch.nn.init.kaiming_normal_(
            torch.empty(*args), a=kwargs.get("a", 0.0), mode=kwargs.get("mode", "fan_in"),
            nonlinearity=kwargs.get("nonlinearity", "leaky_relu"),
        ),
        "rand": lambda *args, **kwargs: torch.rand(*args, **kwargs),
        "ones": lambda *args, **kwargs: torch.ones(*args),
        "unitary": lambda *args, **kwargs: unitary_rn_normal_matrix(*args, **kwargs),
        "orthogonal": lambda *args, **kwargs: torch.nn.init.orthogonal_(
            torch.empty(*args), gain=kwargs.get("gain", 1.0)
        ),
    }
    DEFAULT_FEEDBACKS_STR_NORM_CLIP_VALUE = {
        "randn": 1.0,
        "xavier_normal": torch.inf,
        "kaiming_normal": torch.inf,
        "rand": 1.0,
        "ones": 1.0,
        "unitary": torch.inf,
        "orthogonal": torch.inf,
    }

    def __init__(
            self,
            *,
            params: Optional[Sequence[torch.nn.Parameter]] = None,
            output_params: Optional[Sequence[torch.nn.Parameter]] = None,
            layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
            output_layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]] = None,
            **kwargs
    ):
        """
        Constructor for Eprop class.

        :param params: The hidden parameters to optimize. If not provided, eprop will try to find the hidden parameters
                        by looking for the parameters of the layers provided or in the inputs and hidden layers of the
                        model provided in the trainer.
        :type params: Optional[Sequence[torch.nn.Parameter]]
        :param output_params: The output parameters to optimize. If not provided, eprop will try to find the output
                        parameters by looking for the parameters of the output layers provided or in the output layers
                        of the model provided in the trainer.
        :type output_params: Optional[Sequence[torch.nn.Parameter]]
        :param layers: The hidden layers to optimize. If not provided, eprop will try to find the hidden layers
                        by looking for the layers of the model provided in the trainer.
        :type layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]]
        :param output_layers: The output layers to optimize. If not provided, eprop will try to find the output layers
                        by looking for the layers of the model provided in the trainer.
        :type output_layers: Optional[Union[Sequence[torch.nn.Module], torch.nn.Module]]
        :param kwargs: Keyword arguments to pass to the parent class and to configure the algorithm.

        :keyword Optional[torch.optim.Optimizer] optimizer: The optimizer to use. If provided make sure to provide the
                        param_group in the following format:
                                [{"params": params, "lr": params_lr}, {"params": output_params, "lr": output_params_lr}]
                        The index of the group must be the same as the OPTIMIZER_PARAMS_GROUP_IDX and
                        OPTIMIZER_OUTPUT_PARAMS_GROUP_IDX constants which are 0 and 1 respectively.
                        If not provided, torch.optim.Adam is used.
        :keyword Optional[Union[Dict[str, Union[torch.nn.Module, Callable]], torch.nn.Module, Callable]] criterion: The
                        criterion to use for the output learning signal. If not provided, torch.nn.MSELoss is used. Note
                        that this criterion will be minimized.
        :keyword float params_lr: The learning rate for the hidden parameters. Defaults to 1e-4.
        :keyword float output_params_lr: The learning rate for the output parameters. Defaults to 2e-4.
        :keyword float eligibility_traces_norm_clip_value: The value to clip the eligibility traces norm to.
                        Defaults to torch.inf.
        :keyword float grad_norm_clip_value: The value to clip the gradients norm to. This parameter is used to
                        normalize the gradients of the parameters in order to help the convergence and avoid
                        overflowing. Defaults to 1.0.
        :keyword str feedbacks_gen_strategy: The strategy to use to generate the feedbacks. Defaults to
                        Eprop.DEFAULT_FEEDBACKS_GEN_STRATEGY which is "xavier_normal". The available strategies are
                        stored in Eprop.FEEDBACKS_GEN_FUNCS which are:
                            - "randn": Normal distribution with mean 0 and variance 1.
                            - "xavier_normal": Xavier normal distribution.
                            - "rand": Uniform distribution between 0 and 1.
                            - "unitary": Unitary matrix with normal distribution.
        :keyword float nan: The value to use to replace the NaN values in the gradients. Defaults to 0.0.
        :keyword float posinf: The value to use to replace the inf values in the gradients. Defaults to 1.0.
        :keyword float neginf: The value to use to replace the -inf values in the gradients. Defaults to -1.0.
        :keyword bool save_state: Whether to save the state of the optimizer. Defaults to True.
        :keyword bool load_state: Whether to load the state of the optimizer. Defaults to True.
        :keyword bool raise_non_finite_errors: Whether to raise non-finite errors when detected. Defaults to False.
        """
        kwargs.setdefault("save_state", True)
        kwargs.setdefault("load_state", True)
        kwargs.setdefault("backward_time_steps", 1)
        kwargs.setdefault("optim_time_steps", 1)
        kwargs.setdefault("criterion", torch.nn.MSELoss())
        kwargs.setdefault("alpha", 0.001)
        super().__init__(
            params=params,
            layers=layers,
            output_layers=output_layers,
            **kwargs
        )
        self.output_params = output_params or []
        self._feedbacks_gen_strategy = kwargs.get("feedbacks_gen_strategy", self.DEFAULT_FEEDBACKS_GEN_STRATEGY).lower()
        if self._feedbacks_gen_strategy not in self.FEEDBACKS_GEN_FUNCS:
            raise NotImplementedError(
                f"Feedbacks generation strategy '{self._feedbacks_gen_strategy}' is not implemented."
                f"Maybe you meant one of the following: {', '.join(self.FEEDBACKS_GEN_FUNCS.keys())}"
                f" or you can implement your own by adding it to the Eprop.FEEDBACKS_GEN_FUNCS dictionary."
                f" If your new strategy is a common one, please consider contributing to the library by opening a "
                f"pull request on https://github.com/NeuroTorch/NeuroTorch."
            )
        self.feedback_weights = None
        self.rn_gen = torch.Generator()
        self.rn_gen.manual_seed(kwargs.get("seed", 0))
        self.eligibility_traces = [torch.zeros_like(p) for p in self.params]
        self.output_eligibility_traces = [torch.zeros_like(p) for p in self.output_params]
        self.learning_signals = defaultdict(list)
        self.param_groups = []
        self.eval_criterion = kwargs.get("eval_criterion", self.criterion)
        self.gamma = kwargs.get("gamma", 0.001)
        self._default_params_lr = kwargs.get("params_lr", 1e-5)
        self._default_output_params_lr = kwargs.get("output_params_lr", 2e-5)
        self.eligibility_traces_norm_clip_value = to_tensor(kwargs.get("eligibility_traces_norm_clip_value", torch.inf))
        self.learning_signal_norm_clip_value = to_tensor(kwargs.get("learning_signal_norm_clip_value", torch.inf))
        self.feedback_weights_norm_clip_value = to_tensor(kwargs.get(
            "feedback_weights_norm_clip_value",
            self.DEFAULT_FEEDBACKS_STR_NORM_CLIP_VALUE.get(str(self._feedbacks_gen_strategy), torch.inf)
        ))
        self.raise_non_finite_errors = kwargs.get("raise_non_finite_errors", False)

    def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
        """
        Load the state of the optimizer from the checkpoint.

        :param trainer: The trainer object that is used for training.
        :param checkpoint: The checkpoint dictionary.
        :param kwargs: Additional keyword arguments.

        :return: None
        """
        if self.load_state:
            state = checkpoint.get(self.name, {})
            opt_state_dict = state.get(self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY, None)
            if opt_state_dict is not None:
                saved_param_groups = opt_state_dict["param_groups"]
                if self.optimizer is None:
                    self.optimizer = self.DEFAULT_OPTIMIZER_CLS(saved_param_groups)
                self.optimizer.load_state_dict(opt_state_dict)
                self.param_groups = self.optimizer.param_groups
                self.params = self.param_groups[self.OPTIMIZER_PARAMS_GROUP_IDX]["params"]
                self.output_params = self.param_groups[self.OPTIMIZER_OUTPUT_PARAMS_GROUP_IDX]["params"]
                self.feedback_weights = state.get(self.CHECKPOINT_FEEDBACK_WEIGHTS_KEY, None)

    def get_checkpoint_state(self, trainer, **kwargs) -> object:
        """
        Get the state of the optimizer to be saved in the checkpoint.

        :param trainer: The trainer object that is used for training.
        :param kwargs: Additional keyword arguments.
        :return: The state of the optimizer to be saved in the checkpoint.
        """
        if self.save_state:
            state = {}
            if self.optimizer is not None:
                state[self.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY] = self.optimizer.state_dict()
                state[self.CHECKPOINT_FEEDBACK_WEIGHTS_KEY] = self.feedback_weights
            return state
        return None

    def _get_feedback_gen_func(self):
        """
        Get the feedbacks generation function.

        :return:
        """
        if self._feedbacks_gen_strategy is None:
            self._feedbacks_gen_strategy = self.DEFAULT_FEEDBACKS_GEN_STRATEGY
        if self._feedbacks_gen_strategy not in self.FEEDBACKS_GEN_FUNCS:
            raise NotImplementedError(
                f"Feedbacks generation strategy '{self._feedbacks_gen_strategy}' is not implemented."
                f"Maybe you meant one of the following: {', '.join(self.FEEDBACKS_GEN_FUNCS.keys())}"
                f" or you can implement your own by adding it to the Eprop.FEEDBACKS_GEN_FUNCS dictionary."
                f" If your new strategy is a common one, please consider contributing to the library by opening a "
                f"pull request on https://github.com/NeuroTorch/NeuroTorch."
            )
        return self.FEEDBACKS_GEN_FUNCS[self._feedbacks_gen_strategy]

    def make_feedback_weights(self, *args, **kwargs):
        """
        Generate the feedback weights for each params.
        The random feedback is noted B_{ij} in Bellec's paper :cite:t:`bellec_solution_2020`.

        :return: The feedback weights for each params.
        """
        feedback_gen_func = self._get_feedback_gen_func()
        feedback_weights = feedback_gen_func(*args, **kwargs)
        clip_tensors_norm_(feedback_weights, max_norm=self.feedback_weights_norm_clip_value)
        return feedback_weights

    def initialize_feedback_weights(
            self,
            y_batch: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None
    ) -> Dict[str, List[torch.Tensor]]:
        """
        TODO : Non-random feedbacks must be implemented with {W_out}.T
        Initialize the feedback weights for each params.
        The random feedback is noted B_{ij} in Bellec's paper :cite:t:`bellec_solution_2020`.
        The keys of the feedback_weights dictionary are the names of the output layers.

        :param y_batch: The batch of the target values.

        :return: The feedback weights for each params.
        """
        if y_batch is None:
            y_batch = self.trainer.current_training_state.y_batch
        if not isinstance(y_batch, dict):
            y_batch = {self.DEFAULT_Y_KEY: y_batch}
        last_dims = [p.shape[-1] if p.ndim > 0 else 1 for p in self.params]
        self.feedback_weights = {
            k: [
                self.make_feedback_weights(y_batch_item.shape[-1], pld, generator=self.rn_gen)
                for pld in last_dims
            ] for k, y_batch_item in y_batch.items()
        }
        return self.feedback_weights

    def initialize_params(self, trainer=None):
        """
        Initialize the parameters of the optimizer.

        :Note: Must be called after :meth:`initialize_output_params` and :meth:`initialize_layers`.

        :param trainer: The trainer to use.

        :return: None
        """
        if not self.params and self.optimizer:
            self.params = self.optimizer.param_groups[self.OPTIMIZER_PARAMS_GROUP_IDX]["params"]

        if not self.params:
            self.params = [
                param
                for layer in self.layers
                for param in layer.parameters()
            ]
        if not self.params:
            warnings.warn("No hidden parameters found. Please provide them manually if you have any.")

        return self.params

    def initialize_output_params(self, trainer=None):
        """
        Initialize the output parameters of the optimizer. Try multiple ways to identify the
        output parameters if those are not provided by the user.

        :Note: Must be called after :meth:`initialize_output_layers`.

        :param trainer: The trainer object.

        :return: None
        """
        if not self.output_params and self.optimizer:
            self.output_params = self.optimizer.param_groups[self.OPTIMIZER_OUTPUT_PARAMS_GROUP_IDX]["params"]

        if not self.output_params:
            self.output_params = [
                param
                for layer in self.output_layers
                for param in layer.parameters()
            ]
        if not self.output_params:
            raise ValueError("Could not find output parameters. Please provide them manually.")

        return

    def initialize_output_layers(self, trainer):
        """
        Initialize the output layers of the optimizer. Try multiple ways to identify the output layers if those are not
        provided by the user.

        :Note: Must be called before :meth:`initialize_output_params`.

        :param trainer: The trainer object.
        :return: None.
        """
        if not self.output_layers:
            self.output_layers = []
            possible_attrs = ["output_layers", "output_layer"]
            for attr in possible_attrs:
                obj = getattr(trainer.model, attr, [])
                if isinstance(obj, (Sequence, torch.nn.ModuleList)):
                    obj = list(obj)
                elif isinstance(obj, (Mapping, torch.nn.ModuleDict)):
                    obj = list(obj.values())
                elif isinstance(obj, torch.nn.Module):
                    obj = [obj]
                self.output_layers += list(obj)

        if not self.output_layers:
            raise ValueError(
                "Could not find output layers. Please provide them manually."
            )

    def initialize_layers(self, trainer):
        """
        Initialize the layers of the optimizer. Try multiple ways to identify the output layers if those are not
        provided by the user.

        :param trainer: The trainer object.

        :return: None
        """
        if not self.layers:
            self.layers = []
            possible_attrs = ["input_layers", "input_layer", "hidden_layers", "hidden_layer"]
            for attr in possible_attrs:
                if hasattr(trainer.model, attr):
                    obj = getattr(trainer.model, attr, [])
                    if isinstance(obj, (Sequence, torch.nn.ModuleList)):
                        obj = list(obj)
                    elif isinstance(obj, (Mapping, torch.nn.ModuleDict)):
                        obj = list(obj.values())
                    elif isinstance(obj, torch.nn.Module):
                        obj = [obj]
                    self.layers += list(obj)
        if not self.layers:
            warnings.warn(
                "No hidden layers found. Please provide them manually if you have any."
                "If you are using only one layer, please note that E-prop is equivalent to a TBPTT. If this is the"
                "case, one might consider using TBPTT instead of E-prop."
            )

    def initialize_param_groups(self) -> List[Dict[str, Any]]:
        """
        The learning rate are initialize. If the user has provided a learning rate for each parameter, then it is used.

        :return: the param_groups.
        """
        self.param_groups = []
        list_insert_replace_at(
            self.param_groups,
            self.OPTIMIZER_PARAMS_GROUP_IDX,
            {"params": self.params, "lr": self._default_params_lr}
        )
        list_insert_replace_at(
            self.param_groups,
            self.OPTIMIZER_OUTPUT_PARAMS_GROUP_IDX,
            {"params": self.output_params, "lr": self._default_output_params_lr}
        )
        return self.param_groups

    def eligibility_traces_zeros_(self):
        """
        Set the eligibility traces to zero.

        :return: None
        """
        self.eligibility_traces = [torch.zeros_like(p) for p in self.params]
        self.output_eligibility_traces = [torch.zeros_like(p) for p in self.output_params]

    def start(self, trainer, **kwargs):
        """
        Start the training process with E-prop.

        :param trainer: The trainer object for the training process with E-prop.
        :param kwargs: Additional arguments.

        :return: None
        """
        LearningAlgorithm.start(self, trainer, **kwargs)
        self.initialize_output_layers(trainer)
        self.initialize_output_params(trainer)
        self.initialize_layers(trainer)
        self.initialize_params(trainer)
        zero_grad_params(self.params)
        zero_grad_params(self.output_params)

        if self.criterion is None and trainer.criterion is not None:
            self.criterion = trainer.criterion

        if not self.param_groups:
            self.initialize_param_groups()
        if not self.optimizer:
            self.optimizer = self.create_default_optimizer()

        self._initialize_original_forwards()
        self.eligibility_traces_zeros_()

    def on_batch_begin(self, trainer, **kwargs):
        """
        For each batch. Initialize the random feedback weights if not already done. Also, set the eligibility traces
        to zero.

        :param trainer: The trainer object.
        :param kwargs: Additional arguments.

        :return: None
        """
        super().on_batch_begin(trainer)
        if trainer.model.training:
            if self.feedback_weights is None:
                self.initialize_feedback_weights(trainer.current_training_state.y_batch)
        self.eligibility_traces_zeros_()
        zero_grad_params(self.params)
        zero_grad_params(self.output_params)

    def decorate_forwards(self):
        """
        Ensure that the forward pass is decorated. THe original forward and the hidden layers names are stored. The
        hidden layers forward method are decorated using :meth: `_decorate_hidden_forward`. The output layers forward
        are decorated using :meth: `_decorate_output_forward` from TBPTT.

        Here, we are using decorators to introduce a specific behavior in the forward pass. For E-prop, we need to
        ensure that the gradient is computed and optimize at each time step t of the sequence. This can be achieved by
        decorating our forward. However, we do keep in storage the previous forward pass. This is done to ensure
        that the forward pass is not modified permanently in any way.


        :return: None
        """
        if self.trainer.model.training:
            if not self._forwards_decorated:
                self._initialize_original_forwards()
            self._hidden_layer_names.clear()

            for layer in self.layers:
                self._hidden_layer_names.append(layer.name)
                if self._use_hooks:
                    hook = layer.register_forward_hook(self._hidden_hook, with_kwargs=True)
                    self.forwards_hooks.append(hook)
                else:
                    layer.forward = self._decorate_hidden_forward(layer.forward, layer.name)

            for layer in self.output_layers:
                if self._use_hooks:
                    hook = layer.register_forward_hook(self._output_hook, with_kwargs=True)
                    self.forwards_hooks.append(hook)
                else:
                    layer.forward = self._decorate_forward(layer.forward, layer.name)
            self._forwards_decorated = True

    def _decorate_hidden_forward(self, forward, layer_name: str) -> Callable:
        """
        In TBPTT, we decorate forward to ensure that the backpropagation and the optimizer at t is done for the entire
        network. In E-prop, we want to backpropagate the hidden layers locally (and not the entire network) at each
        time step t.

        :param forward: The forward method to decorate.
        :param layer_name: The name of the layer.

        :return: The decorated forward method
        """
        def _forward(*args, **kwargs):
            out = forward(*args, **kwargs)
            t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
            if t is None:
                return out
            out_tensor, hh = unpack_out_hh(out)
            list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
            if len(self._layers_buffer[layer_name]) >= self.backward_time_steps:
                self._hidden_backward_at_t(t, self.backward_time_steps, layer_name)
                out = recursive_detach(out)
            return out
        return _forward

    def _hidden_hook(self, module, args, kwargs, output) -> None:
        r"""
        In TBPTT, we decorate forward to ensure that the backpropagation and the optimizer at t is done for the entire
        network. In E-prop, we want to backpropagate the hidden layers locally (and not the entire network) at each
        time step t.
        """
        t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
        if t is None:
            return

        layer_name = module.name
        out_tensor, hh = unpack_out_hh(output)
        list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
        if len(self._layers_buffer[layer_name]) >= self.backward_time_steps:
            self._hidden_backward_at_t(t, self.backward_time_steps, layer_name)
            output = recursive_detach_(output)
        return

    def _hidden_backward_at_t(self, t: int, backward_t: int, layer_name: str):
        """
        Here, we compute the eligibility trace as seen in equation (13) from :cite:t:`bellec_solution_2020`. Please
        note that while the notation used in this paper for the equation (13) is [dz/dW]_{local}, we have used [dy/dW]
        in order to be coherent with our own convention.

        :param t: The current time step. For example, if we have a time series of length 100, t will be in [0, 99].
        :param backward_t: The number of time steps to go back in time.
        :param layer_name: The name of the layer.

        :return: None
        """
        pred_batch = torch.squeeze(self._get_pred_batch_from_buffer(layer_name))
        dy_dw_locals = dy_dw_local(y=pred_batch, params=self.params, retain_graph=True, allow_unused=True)
        with torch.no_grad():
            self.eligibility_traces = [
                self.gamma * et + torch.nan_to_num(
                    dy_dw.to(et.device),
                    nan=0.0,
                    neginf=-self.eligibility_traces_norm_clip_value,
                    posinf=self.eligibility_traces_norm_clip_value,
                )
                for et, dy_dw in zip(self.eligibility_traces, dy_dw_locals)
            ]
            clip_tensors_norm_(self.eligibility_traces, self.eligibility_traces_norm_clip_value)
            self._layers_buffer[layer_name].clear()

    def _backward_at_t(self, t: int, backward_t: int, layer_name: str):
        """
        Apply the criterion on the batch. The gradients of each parameters are then updated but are not yet optimized.

        :param t: current time step
        :param backward_t: number of time steps to go back in time
        :param layer_name: name of the layer

        :return: None
        """
        y_batch = self._get_y_batch_slice_from_trainer((t + 1) - backward_t, t + 1, layer_name)
        pred_batch = self._get_pred_batch_from_buffer(layer_name)
        batch_loss = self.apply_criterion(pred_batch, y_batch)
        if batch_loss.grad_fn is None:
            raise ValueError(
                f"batch_loss.grad_fn is None. This is probably an internal error. Please report this issue on GitHub."
            )
        with torch.no_grad():
            errors = self.compute_errors(pred_batch, y_batch)
        output_grads = dy_dw_local(torch.mean(batch_loss), self.output_params, retain_graph=True, allow_unused=True)
        with torch.no_grad():
            self.output_eligibility_traces = [
                self.alpha * et + torch.nan_to_num(
                    dy_dw.to(et.device),
                    nan=0.0,
                    neginf=-self.eligibility_traces_norm_clip_value,
                    posinf=self.eligibility_traces_norm_clip_value,
                )
                for et, dy_dw in zip(self.output_eligibility_traces, output_grads)
            ]
            clip_tensors_norm_(self.output_eligibility_traces, self.eligibility_traces_norm_clip_value)
        self.update_grads(errors)
        with torch.no_grad():
            self._layers_buffer[layer_name].clear()

    def compute_learning_signals(self, errors: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        TODO : Determine if we normalize with the number of output when computing the learning signal : If multiple
        TODO : output layers, do we sum the learning signals or do we average them ? Should we make a 'reduce' param?
        TODO : When averaging, add factor 1/n to the learning signal. It "kind of" results in a change of learning rate.
        The learning signals are computed using equation (28) from :cite:t:`bellec_solution_2020`.

        :param errors: The errors to use to compute the learning signals.

        :return: List of the learning signals for each parameter.
        """
        learning_signals = [torch.zeros((p.shape[-1] if p.ndim > 0 else 1), device=p.device) for p in self.params]
        for k, feedbacks in self.feedback_weights.items():
            if k not in errors:
                raise ValueError(
                    f"This is an internal error. Please report this issue on GitHub."
                    f"Key {k} from {self.feedback_weights.keys()=} not found in errors of keys {errors.keys()}."
                )
            torch.nan_to_num_(errors[k], nan=self.nan, posinf=self.posinf, neginf=self.neginf)
            error_mean = torch.mean(errors[k].view(-1, errors[k].shape[-1]), dim=0).view(1, -1)
            if torch.isfinite(self.learning_signal_norm_clip_value):
                clip_tensors_norm_(error_mean, max_norm=self.learning_signal_norm_clip_value)
            for i, feedback in enumerate(feedbacks):
                learning_signals[i] = learning_signals[i] + torch.matmul(error_mean, feedback.to(error_mean.device))
        return learning_signals

    def compute_errors(
            self,
            pred_batch: Union[Dict[str, torch.Tensor], torch.Tensor],
            y_batch: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        The errors for each output is computed then inserted in a dict for further use. This function check if the
        y_batch and pred_batch are given as a dict or a tensor.

        :param pred_batch: prediction of the network
        :param y_batch: target

        :return: dict of errors
        """
        if isinstance(y_batch, dict) or isinstance(pred_batch, dict):
            if isinstance(y_batch, torch.Tensor):
                y_batch = {k: y_batch for k in pred_batch}
            else:
                raise ValueError(f"y_batch must be a dict or a tensor, not {type(y_batch)}.")
            if isinstance(pred_batch, torch.Tensor):
                pred_batch = {k: pred_batch for k in y_batch}
            else:
                raise ValueError(f"pred_batch must be a dict or a tensor, not {type(pred_batch)}.")
            batch_errors = {
                k: (pred_batch[k] - y_batch[k].to(pred_batch[k].device))
                for k in y_batch
            }
        else:
            batch_errors = {self.DEFAULT_Y_KEY: pred_batch - y_batch.to(pred_batch.device)}
        return batch_errors

    def update_grads(
            self,
            errors: Dict[str, torch.Tensor],
    ):
        """
        The learning signal is computed. The gradients of the parameters are then updated as seen in equation (28)
        from :cite:t:`bellec_solution_2020`.

        :param errors: The errors to use to compute the learning signals.

        :return: None
        """
        learning_signals = self.compute_learning_signals(errors)
        with torch.no_grad():
            for param, ls, et in zip(self.params, learning_signals, self.eligibility_traces):
                if param.requires_grad:
                    param.grad += (ls * et.to(ls.device)).to(param.device).view(param.shape).detach()
                    torch.nan_to_num_(param.grad, nan=self.nan, posinf=self.posinf, neginf=self.neginf)
            if torch.isfinite(self.grad_norm_clip_value):
                torch.nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip_value)

            if self.raise_non_finite_errors:
                if not all([torch.isfinite(p.grad).all() for p in self.params]):
                    raise ValueError(
                        "Non-finite detected in hidden parameters gradients. Try to reduce the learning rate of the hidden "
                        "parameters with the argument `params_lr`."
                    )

        with torch.no_grad():
            for out_param, out_el in zip(self.output_params, self.output_eligibility_traces):
                if out_param.requires_grad:
                    out_param.grad += out_el.to(out_param.device).view(out_param.shape).detach()
                    torch.nan_to_num_(out_param.grad, nan=self.nan, posinf=self.posinf, neginf=self.neginf)
            if torch.isfinite(self.grad_norm_clip_value):
                torch.nn.utils.clip_grad_norm_(self.output_params, self.grad_norm_clip_value)

            if self.raise_non_finite_errors:
                if not all([torch.isfinite(p).all() for p in self.output_params]):
                    raise ValueError(
                        "Non-finite detected in output parameters gradients. Try to reduce the learning rate of the output "
                        "parameters with the argument `output_params_lr`."
                    )

    def _make_optim_step(self, **kwargs):
        """
        Set the gradients and the eligibility traces to zero.

        :param kwargs: Additional arguments.

        :return: None
        """
        super()._make_optim_step(**kwargs)
        with torch.no_grad():
            zero_grad_params(self.output_params)
        if self.raise_non_finite_errors:
            if not all([torch.isfinite(p).all() for p in self.params]):
                raise ValueError(
                    "Non-finite detected in hidden parameters. Try to reduce the learning rate of the hidden "
                    "parameters with the argument `params_lr`."
                )
            if not all([torch.isfinite(p).all() for p in self.output_params]):
                raise ValueError(
                    "Non-finite detected in output parameters. Try to reduce the learning rate of the output "
                    "parameters with the argument `output_params_lr`."
                )

    def on_batch_end(self, trainer, **kwargs):
        """
        Ensure that there is not any remaining gradients in the output parameters. The forward are undecorated and the
        gradients are set to zero. The buffer are also cleared.

        :param trainer: The trainer to use for computation.
        :param kwargs: Additional arguments.

        :return: None
        """
        LearningAlgorithm.on_batch_end(self, trainer)
        if trainer.model.training:
            need_optim_step = False
            for layer_name in self._layers_buffer:
                backward_t = len(self._layers_buffer[layer_name])
                if backward_t > 0:
                    need_optim_step = True
                    if layer_name in self._hidden_layer_names:
                        self._hidden_backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
                    else:
                        self._backward_at_t(self._data_n_time_steps - 1, backward_t, layer_name)
            if need_optim_step:
                self._make_optim_step()
        self.undecorate_forwards()
        self._layers_buffer.clear()
        self.optimizer.zero_grad()
        self.eligibility_traces_zeros_()


