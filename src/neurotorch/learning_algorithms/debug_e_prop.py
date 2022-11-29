"""
Goal: Make e-prop with auto-differentiation.
References paper:
	- https://arxiv.org/pdf/2201.07602.pdf
	- RFLO: https://elifesciences.org/articles/43299#s4
	- Bellec: https://www.biorxiv.org/content/10.1101/738385v3.full.pdf+html
	
References code:
	- https://github.com/ChFrenkel/eprop-PyTorch/blob/main/models.py
	- Bellec: https://github.com/IGITUGraz/eligibility_propagation
"""
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from neurotorch import to_numpy, to_tensor
from neurotorch.dimension import SizeTypes
from neurotorch.metrics import PVarianceLoss
from neurotorch.modules.layers import Linear, LILayer
from neurotorch.utils import unpack_out_hh


class DummyLayer(Linear):
	def __init__(
			self,
			input_size: Optional[SizeTypes] = None,
			output_size: Optional[SizeTypes] = None,
			name: Optional[str] = None,
			device: Optional[torch.device] = None,
			**kwargs
	):
		super().__init__(
			input_size=input_size,
			output_size=output_size,
			name=name,
			device=device,
			**kwargs
		)
		

def dummy_train(targets: torch.Tensor):
	targets = to_tensor(targets)[None, ...]
	layer = LILayer(
		input_size=targets.shape[-1],
		output_size=targets.shape[-1],
		device=targets.device
	).build()
	optimizer = torch.optim.Adam(layer.parameters(), lr=0.1, maximize=True)
	criterion = PVarianceLoss()
	preds = None
	p_bar = tqdm(range(100))
	for i in p_bar:
		output_list, hh_list = [targets[:, 0]], [None]
		for t in range(1, targets.shape[1]):
			out, hh = unpack_out_hh(layer(output_list[t-1], hh_list[t-1]))
			output_list.append(out)
			hh_list.append(hh)
		preds = torch.stack(output_list, dim=1)
		loss = criterion(preds, targets)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		p_bar.set_description(f"Loss: {loss.item():.4f}")
	return torch.squeeze(preds)


if __name__ == '__main__':
	target = torch.stack([torch.sin(torch.linspace(0, 2 * np.pi, 100)) for i in range(2)])
	predictions = dummy_train(target)
	
	fig, ax = plt.subplots()
	ax.plot(to_numpy(target).T, label="target")
	ax.plot(to_numpy(predictions).T, label="predictions")
	ax.legend()
	plt.show()

