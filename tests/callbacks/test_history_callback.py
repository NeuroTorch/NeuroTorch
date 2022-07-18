import unittest
# import pytest

from neurotorch.callbacks import TrainingHistory


class TestHistoryCallback(unittest.TestCase):
	def test_concat(self):
		"""
		Test that the concat method works as expected.
		:return: None
		"""
		hist = TrainingHistory()
		other = {'a': 0, 'b': 1}
		hist.concat(other)
		self.assertEqual(hist['a'], [0], f"{hist['a'] = }, expected [0]")
		self.assertEqual(hist['b'], [1], f"{hist['b'] = }, expected [1]")

		# hist.concat({'a': [1, 2], 'b': [3, 4]})
		self.assertRaises(NotImplementedError, lambda: hist.concat({'a': [1, 2], 'b': [3, 4]}))
		# self.assertEqual(hist['a'], [0, 1, 2], f"{hist['a'] = }, expected [0, 1, 2]")
		# self.assertEqual(hist['b'], [1, 3, 4], f"{hist['b'] = }, expected [1, 3, 4]")

	def test_insert(self):
		"""
		Test that the insert method works as expected.
		:return: None
		"""
		hist = TrainingHistory()
		hist.insert(0, {'a': 0, 'b': 1})
		self.assertEqual(hist['a'], [0])
		self.assertEqual(hist['b'], [1])

		self.assertRaises(NotImplementedError, lambda: hist.insert(0, {'a': [1, 2], 'b': [3, 4]}))
		# hist.insert(0, {'a': [1, 2], 'b': [3, 4]})
		# self.assertEqual(hist['a'], [1, 2, 0])
		# self.assertEqual(hist['b'], [3, 4, 1])

	def test_insert_default(self):
		"""
		Test that the insert method works as expected with a default value.
		:return: None
		"""
		hist = TrainingHistory(default_value=None)
		hist.insert(1, {'a': 0, 'b': 1})
		self.assertEqual(hist['a'], [None, 0], f"{hist['a'] = }, expected [None, 0]")
		self.assertEqual(hist['b'], [None, 1], f"{hist['b'] = }, expected [None, 1]")

		hist.insert(0, {'a': 0, 'b': 1})
		self.assertEqual(hist['a'], [0, 0], f"{hist['a'] = }, expected [0, 0]")
		self.assertEqual(hist['b'], [1, 1], f"{hist['b'] = }, expected [1, 1]")

		hist.insert(3, {'a': 0, 'b': 1})
		self.assertEqual(hist['a'], [0, 0, None, 0], f"{hist['a'] = }, expected [0, 0, None, 0]")
		self.assertEqual(hist['b'], [1, 1, None, 1], f"{hist['b'] = }, expected [1, 1, None, 1]")

		# hist.insert(2, {'a': [1, 1], 'b': [0, 0]}, default=None)
		self.assertRaises(NotImplementedError, lambda: hist.insert(2, {'a': [1, 1], 'b': [0, 0]}))
		# self.assertEqual(hist['a'], [0, 0, 1, 1, None, 0], f"{hist['a'] = }, expected [0, 0, 1, 1, None, 0]")
		# self.assertEqual(hist['b'], [1, 1, 0, 0, None, 1], f"{hist['b'] = }, expected [1, 1, 0, 0, None, 1]")
