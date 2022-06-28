from src.neurotorch.modules.layers import WilsonCowanLayer
import unittest
import torch


class WilsonCowanLayerChild(WilsonCowanLayer):
    def initialize_weights_(self):
        self.forward_weights = torch.nn.Parameter(torch.from_numpy(np.zeros(10, 10)), requires_grad=True)


layer = WilsonCowanLayer(input_size=10, output_size=10, std_weight=1, device="cpu")
layer.build()


print(layer.forward_weights)
print(layer.forward_weights.shape)
print(torch.round(layer.forward_weights.detach().mean()))
print(torch.round(layer.forward_weights.detach().std()))



#assert AlmostEqual(layer.forward_weights.mean(), 0.0)
#self.assertAlmostEqual(layer.forward_weights.std(), 2.0))
