import unittest

from neurotorch.utils import get_meta_str


class TestCollections(unittest.TestCase):
	def test_get_meta_str_int_input(self):
		self.assertEqual(get_meta_str(1), "1")
	
	def test_get_meta_str_list_input(self):
		self.assertEqual(get_meta_str([1, 2, 3]), "1_2_3")
	
	def test_get_meta_str_dict_input(self):
		self.assertEqual(get_meta_str({"a": 1, "b": 2}), "a-1_b-2")
		
	def test_get_meta_str_mixed_input(self):
		self.assertEqual(get_meta_str([{"b": 2, "a": 1}, {1: 2, 3: 4}]), "a-1_b-2_1-2_3-4")
	
	def test_get_meta_str_custom_object_input(self):
		class CustomObject:
			def __repr__(self):
				return "my_repr"
		
		self.assertEqual(
			get_meta_str([{"b": 2, "a": 1}, {1: 2, 3: 4}, 5, 6, 7, CustomObject()]),
			"a-1_b-2_1-2_3-4_5_6_7_my_repr"
		)
	
