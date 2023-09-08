import os
from typing import Optional, Type

import requests
import torch
import tqdm

import neurotorch as nt
from neurotorch.regularization import BaseRegularization


def get_optimizer(optimizer_name: str) -> Type[torch.optim.Optimizer]:
    name_to_opt = {
        "sgd"     : torch.optim.SGD,
        "adam"    : torch.optim.Adam,
        "adamax"  : torch.optim.Adamax,
        "rmsprop" : torch.optim.RMSprop,
        "adagrad" : torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "adamw"   : torch.optim.AdamW,
    }
    return name_to_opt[optimizer_name.lower()]


def get_regularization(
        regularization_name: Optional[str],
        parameters,
        **kwargs
) -> Optional[BaseRegularization]:
    if regularization_name is None or not regularization_name:
        return None
    regs = regularization_name.lower().split('_')
    name_to_reg = {
        "l1"  : nt.L1,
        "l2"  : nt.L2,
        "dale": nt.DaleLaw,
    }
    return nt.RegularizationList([name_to_reg[reg](parameters, **kwargs) for reg in regs])


class GoogleDriveDownloader:
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def __init__(
            self,
            file_id: str,
            dest_path: str,
            *,
            chunk_size: Optional[int] = 32768,
            skip_existing: bool = True,
            verbose: bool = True
    ):
        self.file_id = file_id
        self.dest_path = dest_path
        self.chunk_size = chunk_size
        self.skip_existing = skip_existing
        self.verbose = verbose

    def download(self):
        session = requests.Session()

        response = session.get(self.DOWNLOAD_URL, params={'id': self.file_id}, stream=True)
        token = self.get_confirm_token(response)

        if token:
            params = {'id': self.file_id, 'confirm': token}
            response = session.get(self.DOWNLOAD_URL, params=params, stream=True)

        self.save_response_content(response)

    def save_response_content(self, response):
        if self.skip_existing and os.path.exists(self.dest_path):
            if self.verbose:
                print(f"Skipping '{self.dest_path}' because it already exists.")
            return
        os.makedirs(os.path.dirname(self.dest_path), exist_ok=True)
        with open(self.dest_path, "xb") as f:
            for chunk in tqdm.tqdm(response.iter_content(self.chunk_size), unit='chunk', disable=not self.verbose):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
