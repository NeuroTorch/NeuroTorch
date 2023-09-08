import os.path
from typing import Callable, Optional, Tuple, Iterable

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import gaussian_filter1d
import neurotorch as nt
from tutorials.util import GoogleDriveDownloader


class TimeSeriesDataset(Dataset):
    ROOT_FOLDER = "data/ts/"
    FILE_ID_NAME = {
        "SampleZebrafishData_PaulDeKoninckLab_2020-12-16.npy": "1-3jgAZiNU__NxxhXub7ezAJUqDMFpMCO",
        "Stimulus_data_2022_02_23_fish3_1.npy": "19DnsoI_z4IAWGSLP-32JR4rEMMBMp2Gf",
    }

    def __init__(
            self,
            input_transform: Optional[torch.nn.Module] = None,
            target_transform: Optional[torch.nn.Module] = None,
            n_units: Optional[int] = None,
            units: Optional[Iterable[int]] = None,
            n_time_steps: Optional[int] = None,
            seed: int = 0,
            filename: Optional[str] = None,
            smoothing_sigma: float = 0.0,
            dataset_length: Optional[int] = 1,
            randomize_indexes: bool = False,
            rn_indexes_seed: int = 0,
            download: bool = True,
            **kwargs
    ):
        """
        Create a dataset of time series data.

        :param input_transform: transform to apply to the input data
        :type input_transform: Optional[torch.nn.Module]
        :param target_transform: transform to apply to the target data
        :type target_transform: Optional[torch.nn.Module]
        :param n_units: number of units to use
        :type n_units: Optional[int]
        :param units: indexes of the units to use
        :type units: Optional[Iterable[int]]
        :param n_time_steps: number of time steps to use
        :type n_time_steps: Optional[int]
        :param seed: seed for the random number generator for the units
        :type seed: int
        :param filename: filename of the dataset
        :type filename: Optional[str]
        :param smoothing_sigma: sigma for the gaussian smoothing
        :type smoothing_sigma: float
        :param dataset_length: number of samples to generate
        :type dataset_length: Optional[int]
        :param randomize_indexes: if True, the indexes are randomized else they are equally spaced
        :type randomize_indexes: bool
        :param rn_indexes_seed: seed for the random number generator used to generate the indexes
        :type rn_indexes_seed: int
        :param kwargs: additional arguments
        """
        super().__init__()
        self.kwargs = kwargs
        verbose = kwargs.get("verbose", False)
        self.ROOT_FOLDER = kwargs.get("root_folder", self.ROOT_FOLDER)
        if filename is None:
            filename = list(self.FILE_ID_NAME.keys())[0]
            download = True
        path = os.path.join(self.ROOT_FOLDER, filename)
        if os.path.exists(path):
            download = False
        if download:
            assert filename in self.FILE_ID_NAME, \
                f"File {filename} not found in the list of available files: {list(self.FILE_ID_NAME.keys())}."
            GoogleDriveDownloader(
                self.FILE_ID_NAME[filename], path, skip_existing=True, verbose=verbose
            ).download()
        self.filename = filename
        self.ts = np.load(path)
        self.original_time_series = self.ts.copy()
        if self.kwargs.get("rm_dead_units", True):
            self.ts = self.ts[np.sum(np.abs(self.ts), axis=-1) > 0, :]
        self.total_n_neurons, self.total_n_time_steps = self.ts.shape

        self.seed = seed
        random_generator = np.random.RandomState(seed)

        if units is not None:
            units = list(units)
            assert n_units is None or len(
                units
            ) == n_units, "Number of units and number of units in units must be equal"
            n_units = len(units)
        elif n_units is not None:
            if n_units < 0:
                n_units = self.total_n_neurons + n_units + 1
            units = random_generator.randint(self.total_n_neurons, size=n_units)
        else:
            n_units = self.total_n_neurons
            units = random_generator.randint(self.total_n_neurons, size=n_units)

        if n_time_steps is None or n_time_steps > self.total_n_time_steps or n_time_steps < 0:
            self.n_time_steps = self.total_n_time_steps
        else:
            self.n_time_steps = n_time_steps

        self.n_units = n_units
        self.units_indexes = units
        self.data = self.ts[self.units_indexes].T
        self.sigma = smoothing_sigma

        for neuron in tqdm.tqdm(range(self.data.shape[-1]), desc="Smoothing data", disable=not verbose, unit="unit"):
            if self.sigma > 0.0:
                self.data[:, neuron] = gaussian_filter1d(self.data[:, neuron], sigma=self.sigma)
            self.data[:, neuron] = self.data[:, neuron] - np.min(self.data[:, neuron])
            self.data[:, neuron] = self.data[:, neuron] / (np.max(self.data[:, neuron]) + self.kwargs.get("eps", 1e-5))

        self.data = nt.to_tensor(self.data, dtype=torch.float32)
        self.transform = input_transform
        self.target_transform = target_transform
        self.dataset_length = dataset_length
        if self.dataset_length is None or self.dataset_length <= 0:
            self.dataset_length = max(1, self.total_n_time_steps - self.n_time_steps)
        self.randomize_indexes = randomize_indexes
        self.rn_indexes_seed = rn_indexes_seed
        self.rn_index_generator = np.random.RandomState(rn_indexes_seed)
        self.target_skip_first = kwargs.get("target_skip_first", True)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        if self.randomize_indexes:
            index = self.rn_index_generator.randint(self.total_n_time_steps - self.n_time_steps + 1)
        else:
            index = int((item / len(self)) * (self.total_n_time_steps - self.n_time_steps))
        if self.transform is None:
            t0_transformed = torch.unsqueeze(self.data[index], dim=0)
        else:
            t0_transformed = self.transform(torch.unsqueeze(self.data[index], dim=0))

        target_start_idx = index
        if self.target_skip_first:
            target_start_idx += 1
        target = self.data[target_start_idx:self.n_time_steps + index]
        if self.target_transform is None:
            target_transformed = target
        else:
            target_transformed = self.transform(target)
        return t0_transformed, target_transformed

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}"
        repr_str += f"<{self.filename}>"
        repr_str += f"("
        repr_str += f"n_units={self.n_units}/{self.total_n_neurons}, "
        repr_str += f"n_time_steps={self.n_time_steps}, "
        repr_str += f"dataset_length={self.dataset_length}/{max(1, self.total_n_time_steps - self.n_time_steps)}, "
        repr_str += f"sigma={self.sigma}, "
        repr_str += f"rn_idx={self.randomize_indexes}, "
        # repr_str += f"units={self.units_indexes}"
        repr_str += ")"
        return repr_str

    @property
    def full_time_series(self):
        return self.data[None, :, :]

    @property
    def original_series(self):
        return nt.to_tensor(self.original_time_series, dtype=torch.float32)

    def set_params_from_self(self, kwargs: dict):
        kwargs["n_units"] = self.n_units
        kwargs["n_time_steps"] = self.n_time_steps
        kwargs["smoothing_sigma"] = self.sigma
        kwargs["dataset_length"] = self.dataset_length
        kwargs["filename"] = self.filename
        return kwargs


def get_dataloader(
        filename: str,
        n_workers: Optional[int] = None,
        *args,
        **kwargs
):
    randomize_indexes = kwargs.pop("randomize_indexes", kwargs.pop("dataset_randomize_indexes", False))
    if filename is None or filename.lower().endswith('.npy'):
        dataset = TimeSeriesDataset(filename=filename, randomize_indexes=randomize_indexes, *args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {filename}")

    if kwargs.get("verbose", False):
        print(f"Loaded dataset:\n\t{dataset}\n\tSamples: {len(dataset)}")
    batch_size = min(len(dataset), kwargs.setdefault("batch_size", 32))
    if n_workers is None:
        if len(dataset) == 1:
            n_workers = 0
        else:
            n_workers = 2
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=kwargs.get("shuffle", len(dataset) > 1),
        num_workers=n_workers, pin_memory=kwargs.get("pin_memory", True), persistent_workers=n_workers > 0,
    )