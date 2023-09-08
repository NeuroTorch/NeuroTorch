from collections import defaultdict
from typing import Any, Optional, Dict, Callable, Tuple, Union

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn import metrics as sk_metrics

from .base import BaseMetrics
from ..modules import BaseModel
from ..transforms import to_tensor, to_numpy


class RegressionMetrics(BaseMetrics):
    EPSILON = 1e-12

    @staticmethod
    def get_all_metrics_names_to_func() -> Dict[str, Callable]:
        sep = RegressionMetrics.METRICS_NAMES_SEP
        return {
            f"mae{sep}mean_absolute_error": RegressionMetrics.mean_absolute_error,
            f"mse{sep}mean_squared_error": RegressionMetrics.mean_squared_error,
            f"r2": RegressionMetrics.r2,
            f"d2{sep}d2_tweedie": RegressionMetrics.d2_tweedie,
            f"p_var": RegressionMetrics.p_var,
        }

    @staticmethod
    def compute_y_true_y_pred(
            model: BaseModel,
            dataloader: DataLoader,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
        if device is not None:
            model.to(device)
        model.eval()
        predictions = defaultdict(list)
        targets = defaultdict(list)
        with torch.no_grad():
            for i, (x, y_true) in tqdm(
                    enumerate(dataloader), total=len(dataloader),
                    desc=desc, disable=not verbose, position=p_bar_position,
            ):
                x = x.to(model.device)
                y_true = y_true.to(model.device)
                preds = model.get_prediction_trace(x)
                if isinstance(preds, dict):
                    if not isinstance(targets, dict):
                        targets = {k: targets for k in preds}
                    for k, v in preds.items():
                        predictions[k].extend(to_numpy(v))
                        targets[k].extend(to_numpy(y_true[k]))
                else:
                    predictions["__all__"].extend(to_numpy(preds))
                    targets["__all__"].extend(to_numpy(y_true))
        predictions = {k: np.asarray(v) for k, v in predictions.items()}
        targets = {k: np.asarray(v) for k, v in targets.items()}
        if len(targets) == 1:
            return targets[list(targets.keys())[0]], predictions[list(predictions.keys())[0]]
        return targets, predictions

    @staticmethod
    def mean_absolute_error(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[float, Dict[str, float]]:
        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = RegressionMetrics.compute_y_true_y_pred(
                model=model, dataloader=dataloader, device=device,
                verbose=verbose, desc=desc, p_bar_position=p_bar_position
            )

        if isinstance(y_true, dict):
            return {k: sk_metrics.mean_absolute_error(y_true[k].flatten(), y_pred[k].flatten()) for k in y_true}
        return sk_metrics.mean_absolute_error(y_true.flatten(), y_pred.flatten())

    @staticmethod
    def mean_squared_error(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[float, Dict[str, float]]:
        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = RegressionMetrics.compute_y_true_y_pred(
                model, dataloader, device, verbose, desc, p_bar_position
            )

        if isinstance(y_true, dict):
            return {k: sk_metrics.mean_squared_error(y_true[k].flatten(), y_pred[k].flatten()) for k in y_true}
        return sk_metrics.mean_squared_error(y_true.flatten(), y_pred.flatten())

    @staticmethod
    def r2(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[float, Dict[str, float]]:
        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = RegressionMetrics.compute_y_true_y_pred(
                model, dataloader, device, verbose, desc, p_bar_position
            )

        if isinstance(y_true, dict):
            return {k: sk_metrics.r2_score(y_true[k].flatten(), y_pred[k].flatten()) for k in y_true}
        return sk_metrics.r2_score(y_true.flatten(), y_pred.flatten())

    @staticmethod
    def d2_tweedie(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[float, Dict[str, float]]:
        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = RegressionMetrics.compute_y_true_y_pred(
                model, dataloader, device, verbose, desc, p_bar_position
            )

        if isinstance(y_true, dict):
            return {k: sk_metrics.d2_tweedie_score(y_true[k].flatten(), y_pred[k].flatten()) for k in y_true}
        return sk_metrics.d2_tweedie_score(y_true.flatten(), y_pred.flatten())

    @staticmethod
    def compute_p_var(
            y_true,
            y_pred,
            device: Optional[torch.device] = None,
            reduction: str = 'mean',
    ) -> torch.Tensor:
        y_true, y_pred = to_tensor(y_true), to_tensor(y_pred)
        if device is None:
            device = y_pred.device
        y_true, y_pred = y_true.to(device), y_pred.to(device)
        dims = list(range(y_pred.ndim))[1:]
        mse = torch.mean(mse_loss(y_pred, y_true, reduction='none'), dim=dims)
        var = torch.var(y_true, dim=dims)
        p_var_values = 1 - (mse / (var + RegressionMetrics.EPSILON))
        if reduction.lower() == 'none':
            p_var_value = p_var_values
        elif reduction.lower() == 'mean':
            p_var_value = torch.mean(p_var_values)
        elif reduction.lower() == 'sum':
            p_var_value = torch.sum(p_var_values)
        else:
            raise ValueError(f"Reduction {reduction} not recognized. Try 'none'|'sum'|'mean'.")
        return p_var_value

    @staticmethod
    def p_var(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[float, Dict[str, float]]:
        """
        Compute the p-value of the variance of the prediction.
        :param model: The model to use for computing the p-value.
        :param dataloader: The dataloader to use for computing the p-value.
        :param y_true: The true values.
        :param y_pred: The predicted values.
        :param device: The device to use for computing the p-value.
        :param verbose: Whether to print progress.
        :param desc: The description to use for the progress bar.
        :param p_bar_position: The position of the progress bar.
        :return: The p-value of the variance of the prediction.
        """
        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = RegressionMetrics.compute_y_true_y_pred(
                model=model, dataloader=dataloader, device=device,
                verbose=verbose, desc=desc, p_bar_position=p_bar_position
            )

        if isinstance(y_true, dict):
            return {
                k: to_numpy(RegressionMetrics.compute_p_var(y_true[k], y_pred[k], device))
                for k in y_true
            }
        return to_numpy(RegressionMetrics.compute_p_var(y_true, y_pred, device))

    def __call__(
            self,
            data_loader: DataLoader,
            verbose: Union[bool, int] = False
    ) -> Dict[str, Any]:
        """
        Compute the metrics for the given data_loader.
        :param data_loader: The data loader to use to compute the metrics.
        :param verbose: 0: no progress bar, 1: single progress bar, 2: progress bar for each metrics
                        True: verbose is set to 1, False: verbose is set to 0.
        :return: The metrics computed.
        """
        if isinstance(verbose, bool):
            verbose = 1 if verbose else 0
        assert verbose in [0, 1, 2], "verbose must be 0, 1 or 2"
        output = {}
        p_bar = tqdm(
            enumerate(self.metrics_functions.items()),
            total=len(self.metrics_functions),
            disable=verbose == 0,
            unit="metric",
            position=0,
        )
        self.model.eval()
        y_true, y_pred = RegressionMetrics.compute_y_true_y_pred(
            model=self.model,
            dataloader=data_loader,
            device=self.device,
            verbose=verbose == 2,
            desc="compute_y_true_y_preds",
            p_bar_position=1
        )
        for i, (metric_name, metric_func) in p_bar:
            output[metric_name] = metric_func(
                y_true=y_true,
                y_pred=y_pred,
                device=self.device,
                verbose=verbose == 2,
                desc=metric_name,
                p_bar_position=i + 2
            )
        return output
