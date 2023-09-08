from typing import Union, Dict

import torch


def format_pred_batch(
        raw_pred_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        y_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
):
    """
    This function format the raw pred batch to the same format as y_batch. For example, if y_batch is a dict, then
    raw_pred_batch will be converted to a dict. If raw_pred_batch is a tuple or a list, then raw_pred_batch is
    consider to be in the following format : pred, hidden_state. In this case, only pred will be taken.

    :param raw_pred_batch:
    :param y_batch:
    :return:
    """
    from .collections import maybe_unpack_singleton_dict

    if isinstance(raw_pred_batch, (tuple, list)):
        pred_batch = raw_pred_batch[0]
    elif isinstance(raw_pred_batch, (torch.Tensor, dict)):
        pred_batch = raw_pred_batch
    else:
        raise ValueError(f"Unsupported output type: {type(raw_pred_batch)}")

    if isinstance(y_batch, dict):
        if len(y_batch) == 1 and not isinstance(pred_batch, dict):
            pred_batch = {k: pred_batch for k in y_batch}
        assert isinstance(pred_batch, dict) and isinstance(y_batch, dict), \
            "If y_batch is a dict, pred must be a dict too."
        assert set(pred_batch.keys()) == set(y_batch.keys()), \
            "Keys of y_batch and pred_batch must be the same."
    else:
        if isinstance(pred_batch, dict):
            assert len(pred_batch) == 1, \
                "pred_batch must have only one key if y_batch is not a dict."
            pred_batch = maybe_unpack_singleton_dict(pred_batch)
    return pred_batch
