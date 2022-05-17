import enum
from typing import NamedTuple, Union


class DimensionProperty(enum.Enum):
	NONE = 0
	TIME = 1
	SPATIAL = 2
	
	
class Dimension(NamedTuple):
	size: int
	dtype: DimensionProperty
	
	@staticmethod
	def from_int(size: int) -> "Dimension":
		return Dimension(size, DimensionProperty.NONE)
	
	@staticmethod
	def from_int_or_dimension(dimension: Union[int, "Dimension"]) -> "Dimension":
		if isinstance(dimension, int):
			return Dimension.from_int(dimension)
		return dimension


