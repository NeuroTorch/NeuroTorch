from collections import defaultdict
from typing import Any, Optional, Dict, Callable, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .base import BaseMetrics
from ..modules import BaseModel


class ClassificationMetrics(BaseMetrics):
    """
    This class is used to compute metrics for a classification model.
    """

    @staticmethod
    def get_all_metrics_names_to_func() -> Dict[str, Callable]:
        sep = ClassificationMetrics.METRICS_NAMES_SEP
        return {
            f"accuracy{sep}acc": ClassificationMetrics.accuracy,
            f"precision": ClassificationMetrics.precision,
            f"recall": ClassificationMetrics.recall,
            f"f1": ClassificationMetrics.f1,
            # f"auc": ClassificationMetrics.auc,
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
            for i, (inputs, classes) in tqdm(
                    enumerate(dataloader), total=len(dataloader),
                    desc=desc, disable=not verbose, position=p_bar_position,
            ):
                if inputs.is_sparse:
                    inputs = inputs.to_dense()
                if classes.is_sparse:
                    classes = classes.to_dense()
                inputs = inputs.to(model.device)
                classes = classes.to(model.device)
                outputs = model.get_prediction_proba(inputs, re_outputs_trace=False, re_hidden_states=False)
                if isinstance(outputs, dict):
                    if not isinstance(classes, dict):
                        classes = {k: classes for k in outputs}
                    for k, v in outputs.items():
                        _, preds = torch.max(v, -1)
                        preds = preds.cpu().numpy()
                        predictions[k].extend(preds)
                        if len(classes[k].shape) > 1:
                            classes[k] = torch.argmax(classes[k], -1).view(-1)
                        targets[k].extend(classes[k].cpu().numpy())
                else:
                    _, preds = torch.max(outputs, -1)
                    preds = preds.cpu().numpy()
                    if len(classes.shape) > 1:
                        classes = torch.argmax(classes, -1).view(-1)
                    predictions["__all__"].extend(preds)
                    targets["__all__"].extend(classes.cpu().numpy())
        predictions = {k: np.asarray(v) for k, v in predictions.items()}
        targets = {k: np.asarray(v) for k, v in targets.items()}
        if len(targets) == 1:
            return targets[list(targets.keys())[0]], predictions[list(predictions.keys())[0]]
        return targets, predictions

    @staticmethod
    def accuracy(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[float, Dict[str, float]]:
        from sklearn.metrics import accuracy_score
        assert (y_true is None) == (y_pred is None)

        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = ClassificationMetrics.compute_y_true_y_pred(
                model, dataloader, device, verbose, desc, p_bar_position
            )

        if isinstance(y_true, dict):
            return {k: accuracy_score(y_true[k], y_pred[k]) for k in y_true}
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[float, Dict[str, float]]:
        from sklearn.metrics import precision_score
        assert (y_true is None) == (y_pred is None)

        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = ClassificationMetrics.compute_y_true_y_pred(
                model, dataloader, device, verbose, desc, p_bar_position
            )
        if isinstance(y_true, dict):
            average = {k: ("macro" if len(set(v)) > 2 else "binary") for k, v in y_true.items()}
        else:
            average = "macro" if len(set(y_true)) > 2 else "binary"
        if isinstance(y_true, dict):
            return {k: precision_score(y_true[k], y_pred[k], average=average, zero_division=0) for k in y_true}
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def confusion_matrix(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        from sklearn.metrics import confusion_matrix
        assert (y_true is None) == (y_pred is None)

        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = ClassificationMetrics.compute_y_true_y_pred(
                model, dataloader, device, verbose, desc, p_bar_position
            )

        if isinstance(y_true, dict):
            return {k: confusion_matrix(y_true[k], y_pred[k]) for k in y_true}
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def f1(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[float, Dict[str, float]]:
        from sklearn.metrics import f1_score
        assert (y_true is None) == (y_pred is None)

        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = ClassificationMetrics.compute_y_true_y_pred(
                model, dataloader, device, verbose, desc, p_bar_position
            )

        if isinstance(y_true, dict):
            average = {k: ("macro" if len(set(v)) > 2 else "binary") for k, v in y_true.items()}
        else:
            average = "macro" if len(set(y_true)) > 2 else "binary"

        if isinstance(y_true, dict):
            return {k: f1_score(y_true[k], y_pred[k], average=average, zero_division=0) for k in y_true}
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def recall(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[float, Dict[str, float]]:
        from sklearn import metrics as sk_metrics
        assert (y_true is None) == (y_pred is None)

        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = ClassificationMetrics.compute_y_true_y_pred(
                model, dataloader, device, verbose, desc, p_bar_position
            )

        if isinstance(y_true, dict):
            average = {k: ("macro" if len(set(v)) > 2 else "binary") for k, v in y_true.items()}
        else:
            average = "macro" if len(set(y_true)) > 2 else "binary"

        if isinstance(y_true, dict):
            return {k: sk_metrics.recall_score(y_true[k], y_pred[k], average=average, zero_division=0) for k in y_true}
        return sk_metrics.recall_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def auc(
            model: Optional[BaseModel] = None,
            dataloader: Optional[DataLoader] = None,
            y_true: Optional[np.ndarray] = None,
            y_pred: Optional[np.ndarray] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            desc: Optional[str] = None,
            p_bar_position: int = 0,
    ) -> Union[float, Dict[str, float]]:
        from sklearn import metrics as sk_metrics
        assert (y_true is None) == (y_pred is None)

        if y_true is None:
            assert y_pred is None
            assert model is not None, "Either model or y_pred and y_true must be supplied."
            assert dataloader is not None, "Either model or y_pred and y_true must be supplied."
            y_true, y_pred = ClassificationMetrics.compute_y_true_y_pred(
                model, dataloader, device, verbose, desc, p_bar_position
            )

        if isinstance(y_true, dict):
            rocs = {k: sk_metrics.roc_curve(y_true[k], y_pred[k]) for k in y_true}
            return {k: sk_metrics.auc(rocs[k][0], rocs[k][1]) for k in rocs}

        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_pred)
        return sk_metrics.auc(fpr, tpr)

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
        y_true, y_pred = ClassificationMetrics.compute_y_true_y_pred(
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
