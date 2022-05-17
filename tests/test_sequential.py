import unittest
from src.neurotorch.modules import SequentialModel, BaseLayer
from src.neurotorch import Dimension, DimensionProperty


class TestSequential(unittest.TestCase):
	def test_sequential_init(self):
		model_name = "input"
		model = SequentialModel(
			layers=[
				BaseLayer(Dimension(10, DimensionProperty.NONE), 2, name=model_name),
			]
		)
		self.assertEqual(len(model.input_layers), 0)
		self.assertEqual(len(model.hidden_layers), 0)
		self.assertEqual(len(model.output_layers), 1)
		self.assertEqual(model_name in model.output_layers, True)
		self.assertEqual(model.output_layers[model_name].name, model_name)








