import enum
from typing import Iterable, List, Optional, Union


class DimensionProperty(enum.Enum):
    """
    Enum for dimension properties.

    NONE: No dimension property. This type of dimension can be used for features, neurons, unknown, etc.
    TIME: Time dimension. This type of dimension can be used for time series.
    SPATIAL: Spatial dimension. This type of dimension can be used for spatial data like images, videos, etc.
    """
    NONE = 0
    TIME = 1
    SPATIAL = 2


class Dimension:
    """
    This object is used to represent a dimension.

    Attributes:
        size (int): The size of the dimension.
        dtype (DimensionProperty): The type of the dimension.
        name (str): The name of the dimension.

    """

    def __init__(
            self,
            size: Optional[int] = None,
            dtype: DimensionProperty = DimensionProperty.NONE,
            name: str = None,
    ):
        """
        Constructor for Dimension.

        :param size: The size of the dimension.
        :type size: int
        :param dtype: The type of the dimension.
        :type dtype: DimensionProperty
        :param name: The name of the dimension.
        :type name: str
        """
        self.size: Optional[int] = size
        self.dtype: DimensionProperty = dtype
        self.name: str = name if name is not None else dtype.name

    def __eq__(self, other: Union[int, 'Dimension']) -> bool:
        """
        Check if the dimension is equal to the other dimension. Two dimensions are considered equal if they have the
        same size and type.

        :param other: The other dimension.
        :type other: int or Dimension

        :return: True if the dimensions are equal, False otherwise.
        :rtype: bool
        """
        if other is None:
            return False
        if isinstance(other, int):
            return self == self.from_int(other)
        return self.size == other.size and self.dtype == other.dtype

    def __int__(self) -> int:
        return self.size

    def __str__(self) -> str:
        return f"{self.name}:[{self.size}, {self.dtype.name}]"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_int(size: Optional[int]) -> "Dimension":
        """
        Create a Dimension from an integer.

        :param size: The size of the dimension.
        :type size: int

        :return: A dimension with the given size and None as dtype.
        :rtype: Dimension
        """
        return Dimension(size, DimensionProperty.NONE)

    @staticmethod
    def from_int_or_dimension(dimension: Optional[Union[int, "Dimension"]]) -> "Dimension":
        """
        Create a Dimension from an integer or a Dimension.

        :param dimension: The dimension to convert.
        :type dimension: int or Dimension

        :return: A dimension with the given size and None as dtype if the input is an integer and the given dimension
        if it is a Dimension.
        :rtype: Dimension
        """
        if isinstance(dimension, int) or dimension is None:
            return Dimension.from_int(dimension)
        return dimension


DimensionLike = Union[int, Dimension]


class Size:
    """
    This object is used to represent the size of a space.

    :Attributes:
        dimensions (List[Dimension]): The dimensions of the space.
    """

    def __init__(self, dimensions: Union[int, Dimension, Iterable[Union[int, Dimension]]]):
        """
        Constructor for Size.

        :param dimensions: The dimensions of the space.
        :type dimensions: int or Dimension or Iterable[int or Dimension]
        """
        if isinstance(dimensions, (int, Dimension)):
            dimensions = [dimensions]
        else:
            dimensions = list(dimensions)
        self.dimensions: List[Dimension] = [
            Dimension.from_int_or_dimension(dimension)
            for dimension in dimensions
        ]

    def __str__(self) -> str:
        _str = "Size[" + ", ".join([f"{i}:" + str(dim) for i, dim in enumerate(self.dimensions)])
        return _str[:-2] + "]"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: 'Size') -> bool:
        if other is None:
            return False
        return self.dimensions == other.dimensions

    def __getitem__(self, item):
        return self.dimensions[item]

    def __len__(self) -> int:
        return len(self.dimensions)

    def __iter__(self) -> Iterable[Dimension]:
        return iter(self.dimensions)


SizeTypes = Union[int, Dimension, Iterable[Union[int, Dimension]], Size]
DimensionsLike = Union[Dimension, Iterable[Dimension], Size]
