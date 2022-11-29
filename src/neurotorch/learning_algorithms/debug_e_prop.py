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
import torch
import numpy as np
import matplotlib.pyplot as plt

from neurotorch import to_numpy

target = torch.sin(torch.linspace(0, 2 * np.pi, 100))


fig, ax = plt.subplots()
ax.plot(to_numpy(target), label="target")
ax.legend()
plt.show()

