import enum
from typing import NamedTuple, Union, Optional


class DimensionProperty(enum.Enum):
	NONE = 0
	TIME = 1
	SPATIAL = 2
	

class Dimension(NamedTuple):
	size: Optional[int]
	dtype: DimensionProperty
	
	def __int__(self):
		return self.size
	
	def __str__(self):
		return f"{self.dtype.name}:{self.size}"
	
	@staticmethod
	def from_int(size: Optional[int]) -> "Dimension":
		return Dimension(size, DimensionProperty.NONE)
	
	@staticmethod
	def from_int_or_dimension(dimension: Optional[Union[int, "Dimension"]]) -> "Dimension":
		if isinstance(dimension, int) or dimension is None:
			return Dimension.from_int(dimension)
		return dimension


