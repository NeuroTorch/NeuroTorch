import unittest

from neurotorch.utils import unpack_tuple


class TestUnpackTuple(unittest.TestCase):

    def test_unpack_tuple_fill_none_left(self):
        x = (1, 2)
        expected_length = 3
        expected_output = (None, 1, 2)
        self.assertEqual(
            unpack_tuple(x, expected_length, fill_value=None, fill_method="left"),
            expected_output
        )

    def test_unpack_tuple_fill_none_right(self):
        x = (1, 2)
        expected_length = 3
        expected_output = (1, 2, None)
        self.assertEqual(
            unpack_tuple(x, expected_length, fill_value=None, fill_method="right"),
            expected_output
        )

    def test_unpack_tuple_fill_none_middle(self):
        x = (1, 2)
        expected_length = 3
        expected_output = (1, None, 2)
        self.assertEqual(
            unpack_tuple(x, expected_length, fill_value=None, fill_method="middle"),
            expected_output
        )

    def test_unpack_tuple_fill_none_middle_odd_length(self):
        x = (1, 2, 3)
        expected_length = 4
        expected_output = (1, None, 2, 3)
        self.assertEqual(
            unpack_tuple(x, expected_length, fill_value=None, fill_method="middle"),
            expected_output
        )

    def test_unpack_tuple_fill_value_left(self):
        x = (1, 2)
        expected_length = 3
        expected_output = (3, 1, 2)
        self.assertEqual(
            unpack_tuple(x, expected_length, fill_value=3, fill_method="left"),
            expected_output
        )

    def test_unpack_tuple_aggregate_tuple_left(self):
        x = (1, 2, 3, 4)
        expected_length = 3
        expected_output = ((1, 2), 3, 4)
        self.assertEqual(
            unpack_tuple(x, expected_length, aggregate_type=tuple, aggregate_method="left"),
            expected_output
        )

    def test_unpack_tuple_aggregate_list_left(self):
        x = (1, 2, 3, 4)
        expected_length = 3
        expected_output = ([1, 2], 3, 4)
        self.assertEqual(
            unpack_tuple(x, expected_length, aggregate_type=list, aggregate_method="left"),
            expected_output
        )

    def test_unpack_tuple_aggregate_tuple_right(self):
        x = (1, 2, 3, 4)
        expected_length = 3
        expected_output = (1, 2, (3, 4))
        self.assertEqual(
            unpack_tuple(x, expected_length, aggregate_type=tuple, aggregate_method="right"),
            expected_output
        )

    def test_unpack_tuple_aggregate_list_right(self):
        x = (1, 2, 3, 4)
        expected_length = 3
        expected_output = (1, 2, [3, 4])
        self.assertEqual(
            unpack_tuple(x, expected_length, aggregate_type=list, aggregate_method="right"),
            expected_output
        )

    def test_unpack_tuple_aggregate_tuple_middle(self):
        x = (1, 2, 3, 4)
        expected_length = 3
        expected_output = (1, (2, 3), 4)
        output = unpack_tuple(x, expected_length, aggregate_type=tuple, aggregate_method="middle")
        self.assertEqual(
            expected_output,
            output,
            f"Expected output: {expected_output}, got: {output}"
        )

    def test_unpack_tuple_aggregate_tuple_middle_odd_length(self):
        x = (1, 2, 3, 4, 5)
        expected_length = 4
        expected_output = (1, (2, 3), 4, 5)
        output = unpack_tuple(x, expected_length, aggregate_type=tuple, aggregate_method="middle")
        self.assertEqual(
            expected_output,
            output,
            f"Expected output: {expected_output}, got: {output}"
        )

    def test_unpack_tuple_aggregate_list_middle(self):
        x = (1, 2, 3, 4)
        expected_length = 3
        expected_output = (1, [2, 3], 4)
        self.assertEqual(
            unpack_tuple(x, expected_length, aggregate_type=list, aggregate_method="middle"),
            expected_output
        )


