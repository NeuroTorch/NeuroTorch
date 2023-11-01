import collections.abc
import hashlib
import pickle
from typing import Callable, Dict, List, Any, Union, Sequence, Type, Tuple

import torch


def get_meta_name(params: Dict[str, Any]):
    meta_name = f""
    keys = sorted(list(params.keys()))
    for k in keys:
        meta_name += f"{k}-{params[k]}_"
    return meta_name[:-1]


def hash_params(params: Dict[str, Any]):
    """
    Hash the parameters to get a unique and persistent id.

    Note: This is the old version of hash_params. It is kept for compatibility with old code.
        Please use hash_dict instead which is more general and offers more options.

    :param params: The parameters to hash.

    :return: The hash of the parameters.
    """
    return int(hashlib.md5(get_meta_name(params).encode('utf-8')).hexdigest(), 16)


def get_meta_str(__obj: Any) -> str:
    """
    Get the meta string of an object. The meta string is a string representation of the object
    that can be used as a file name. All mappings are sorted by keys before being converted to
    strings.

    :param __obj: The object to get the meta string of.

    :return: The meta string of the object.

    Examples:
        >>> get_meta_str(1)
        '1'
        >>> get_meta_str([1, 2, 3])
        '1_2_3'
        >>> get_meta_str({"a": 1, "b": 2})
        'a-1_b-2'
        >>> get_meta_str([{"b": 2, "a": 1}, {1: 2, 3: 4}])
        'a-1_b-2_1-2_3-4'
        >>> class CustomObject:
        ...     def __repr__(self):
        ...         return "my_repr"
        ... get_meta_str([{"b": 2, "a": 1}, {1: 2, 3: 4}, 5, 6, 7, CustomObject()])
        'a-1_b-2_1-2_3-4_5_6_7_my_repr'
    """
    meta_name = f""
    if isinstance(__obj, collections.abc.Mapping):
        keys = sorted(list(__obj.keys()))
        meta_name = '_'.join([f"{k}-{get_meta_str(__obj[k])}" for k in keys])
    elif isinstance(__obj, collections.abc.Iterable):
        meta_name = '_'.join([get_meta_str(k) for k in __obj])
    else:
        meta_name = str(repr(__obj))
    return meta_name


def hash_meta_str(
        __obj: Any,
        hash_mth: str = "md5",
        out_type: str = "hex"
) -> Union[str, int]:
    """
    Hash an object to get a unique and persistent id. The hash is computed by hashing the
    string representation of the entry. The string representation is obtained using the function
    `get_meta_str`.

    :param __obj: The object to hash.
    :param hash_mth: The hash method to use. Must be in hashlib.algorithms_available. Default is "md5".
    :param out_type: The type of the output. Must be in ["hex", "int"]. Default is "hex".

    :return: The hash of the object.
    """
    available_out_type = ["hex", "int"]
    import hashlib

    if hash_mth not in hashlib.algorithms_available:
        raise ValueError(f"hash_mth must be in {hashlib.algorithms_available}")
    hash_func = getattr(hashlib, hash_mth, hashlib.md5)
    if out_type == "hex":
        return hash_func(get_meta_str(__obj).encode('utf-8')).hexdigest()
    elif out_type == "int":
        return int(hash_func(get_meta_str(__obj).encode('utf-8')).hexdigest(), 16)
    raise ValueError(f"out_type must be in {available_out_type}")


def save_params(params: Dict[str, Any], save_path: str):
    """
    Save the parameters in a file.

    :param save_path: The path to save the parameters.
    :param params: The parameters to save.

    :return: The path to the saved parameters.
    """
    pickle.dump(params, open(save_path, "wb"))
    return save_path


def get_all_params_combinations(params_space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all possible combinations of parameters.

    :param params_space: Dictionary of parameters.

    :return: List of dictionaries of parameters.
    """
    import itertools
    # get all the combinaison of the parameters
    all_params = list(params_space.keys())
    all_params_values = list(params_space.values())
    all_params_combinaison = list(map(lambda x: list(x), list(itertools.product(*all_params_values))))

    # create a list of dict of all the combinaison
    all_params_combinaison_dict = list(map(lambda x: dict(zip(all_params, x)), all_params_combinaison))
    return all_params_combinaison_dict


def list_of_callable_to_sequential(callable_list: List[Callable]) -> torch.nn.Sequential:
    """
    Convert a list of callable to a list of modules.

    :param callable_list: List of callable.

    :return: List of modules.
    """
    from neurotorch.transforms.wrappers import CallableToModuleWrapper
    return torch.nn.Sequential(
        *[
            c if isinstance(c, torch.nn.Module) else CallableToModuleWrapper(c)
            for c in callable_list
        ]
    )


def sequence_get(__sequence: Sequence, idx: int, default: Any = None) -> Any:
    try:
        return __sequence[idx]
    except IndexError:
        return default


def list_insert_replace_at(__list: List, idx: int, value: Any):
    """
    Insert a value at a specific index. If there is already a value at this index, replace it.

    :param __list: The list to modify.
    :param idx: The index to insert the value.
    :param value: The value to insert.
    """
    if idx < len(__list):
        __list[idx] = value
    else:
        __list.extend([None] * (idx - len(__list)))
        __list.append(value)


def unpack_out_hh(out):
    """
    Unpack the output of a recurrent network.

    :param out: The output of a recurrent network.

    :return: The output of the recurrent network with the hidden state.
                If there is no hidden state, consider it as None.
    """
    out_tensor, hh = None, None
    if isinstance(out, (tuple, list)):
        if len(out) == 2:
            out_tensor, hh = out
        elif len(out) == 1:
            out_tensor = out[0]
        elif len(out) > 2:
            out_tensor, *hh = out
    else:
        out_tensor = out

    return out_tensor, hh


def unpack_x_hh_y(__input) -> Tuple[Any, Any, Any]:
    """
    Unpack the input, hidden state and target of a recurrent network.

    :param __input: The input to unpack.

    :return: The input of the recurrent network with the hidden state and the target.
                If there is no hidden state, consider it as None.
    """
    x, hh, y = None, None, None
    if isinstance(__input, (tuple, list)):
        if len(__input) == 3:
            x, hh, y = __input
        elif len(__input) == 2:
            x, y = __input
        elif len(__input) > 3:
            x, *hh, y = __input
        elif len(__input) == 1:
            x = __input[0]
    else:
        x = __input

    return x, hh, y


def unpack_tuple(
        x: Any,
        expected_length: int,
        fill_value: Any = None,
        fill_method: str = "middle",
        aggregate_method: str = "middle",
        aggregate_type: Type = tuple,
) -> tuple:
    """
    Unpack a tuple of a specific length. If the tuple has a different length, one of the following
    output is returned:
        - If the input 'x' has the expected length, the input 'x' unpacked is returned.
        - If the input 'x' has a length lower than the expected length, the input 'x' unpacked is
            returned with 'fill_value' appended to it using the 'fill_method'.
        - If the input 'x' has a length greater than the expected length, the input 'x' unpacked is
            returned with the elements packed together using the 'aggregate_method'.

    :param x: The tuple to unpack.
    :type x: Any
    :param expected_length: The expected length of the tuple.
    :type expected_length: int
    :param fill_value: The value to fill the tuple with if the length of the tuple is lower than the
        expected length. Default is None.
    :type fill_value: Any
    :param fill_method: The method to use to fill the tuple with 'fill_value's if the length of the tuple is lower than
        the expected length. Must be in ["left", "right", "middle"]. Default is "middle".
    :type fill_method: str
    :param aggregate_method: The method to use to aggregate the tuple if the length of the tuple is
        greater than the expected length. Must be in ["left", "right", "middle"]. Default is "middle".
    :type aggregate_method: str
    :param aggregate_type: The type of the aggregated items if the length of the tuple is greater than the
        expected length. Generally one of [tuple, list]. Default is tuple.

    :return: The unpacked tuple.
    :rtype: tuple

    TODO: Optimize this function.
    """
    output = [fill_value for _ in range(expected_length)]
    x_length = len(x)
    if x_length == expected_length:
        output = tuple(x)
    elif x_length < expected_length:
        if fill_method == "left":
            output[-x_length:] = list(x)
        elif fill_method == "right":
            output[:x_length] = list(x)
        elif fill_method == "middle":
            indexes = list(range(expected_length))
            n_fill = expected_length - x_length
            idx_start_fill = int(x_length / 2)
            idx_end_fill = idx_start_fill + n_fill - 1
            x_indexes = list(set(indexes) - set(range(idx_start_fill, idx_end_fill + 1)))
            for i, idx in enumerate(x_indexes):
                output[idx] = x[i]
        else:
            raise ValueError(f"fill_method must be in ['left', 'right', 'middle']. Got {fill_method}.")
    elif x_length > expected_length:
        n_aggregate = x_length - expected_length + 1
        if aggregate_method == "left":
            output = [aggregate_type(x[:n_aggregate])] + list(x[n_aggregate:])
        elif aggregate_method == "right":
            output = list(x[:-n_aggregate]) + [aggregate_type(x[-n_aggregate:])]
        elif aggregate_method == "middle":
            idx_start_aggregate = int(n_aggregate / 2)
            idx_end_aggregate = idx_start_aggregate + n_aggregate
            output = (
                    list(x[:idx_start_aggregate])
                    + [aggregate_type(x[idx_start_aggregate:idx_end_aggregate])]
                    + list(x[idx_end_aggregate:])
            )
        else:
            raise ValueError(f"aggregate_method must be in ['left', 'right', 'middle']. Got {aggregate_method}.")
    return tuple(output)


def unpack_singleton_dict(x: dict) -> Any:
    """
    Unpack a dictionary with a single key and value. If the dict has more than one key, a ValueError is raised.
    :param x:
    :return:
    """
    if len(x) > 1:
        raise ValueError("x must have a length of zero or one.")
    elif len(x) == 0:
        return None
    return x[list(x.keys())[0]]


def maybe_unpack_singleton_dict(x: Union[dict, Any]) -> Any:
    """
    Accept a dict or any other type. If x is a dict with one key and value, the singleton is unpacked. Otherwise, x is
    returned without being changed.
    :param x:
    :return:
    """
    if isinstance(x, dict) and len(x) <= 1:
        return unpack_singleton_dict(x)
    return x


def mapping_update_recursively(d, u):
    """
    from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    :param d: mapping item that wil be updated
    :param u: mapping item updater

    :return: updated mapping recursively
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = mapping_update_recursively(d.get(k, {}), v)
        else:
            d[k] = v
    return d
