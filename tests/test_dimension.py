import unittest
# import pytest
from neurotorch import Dimension, DimensionProperty, Size


# @pytest.fixture
# def dimension_init(*args, **kwargs) -> Dimension:
# 	return Dimension(*args, **kwargs)
#
#
# @pytest.mark.parametrize(
# 	"dimension_init, expected_attributes",
# 	[
# 		((), (None, DimensionProperty.NONE, DimensionProperty.NONE.name)),
# 	]
# )
# def test_dimension_init(dimension_init, expected_attributes):
# 	assert dimension_init.size == expected_attributes[0]
# 	assert dimension_init.dtype == expected_attributes[1]
# 	assert dimension_init.name == expected_attributes[2]


class TestDimension(unittest.TestCase):
    def test_default_constructor(self):
        dim = Dimension()
        self.assertIs(dim.size, None)
        self.assertEqual(dim.dtype, DimensionProperty.NONE)
        self.assertEqual(dim.name, dim.dtype.name)

    def test_constructor(self):
        dim = Dimension(10)
        self.assertEqual(dim.size, 10)
        self.assertEqual(dim.dtype, DimensionProperty.NONE)
        self.assertEqual(dim.name, dim.dtype.name)

        dim = Dimension(10, DimensionProperty.NONE)
        self.assertEqual(dim.size, 10)
        self.assertEqual(dim.dtype, DimensionProperty.NONE)
        self.assertEqual(dim.name, dim.dtype.name)

        dim = Dimension(10, DimensionProperty.TIME)
        self.assertEqual(dim.size, 10)
        self.assertEqual(dim.dtype, DimensionProperty.TIME)
        self.assertEqual(dim.name, dim.dtype.name)

        dim = Dimension(10, DimensionProperty.SPATIAL)
        self.assertEqual(dim.size, 10)
        self.assertEqual(dim.dtype, DimensionProperty.SPATIAL)
        self.assertEqual(dim.name, dim.dtype.name)

        dim = Dimension(10, DimensionProperty.SPATIAL)
        self.assertEqual(dim.size, 10)
        self.assertEqual(dim.dtype, DimensionProperty.SPATIAL)
        self.assertEqual(dim.name, dim.dtype.name)



