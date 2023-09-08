import enum
from typing import Optional

from .base import (
    BaseLayer,
    BaseNeuronsLayer,
)

from .classical import (
    Linear,
    LinearRNN,
)

from .spiking import (
    LIFLayer,
    ALIFLayer,
    SpyLIFLayer,
    SpyALIFLayer,
    IzhikevichLayer,
)

from .spiking_lpf import (
    SpyLIFLayerLPF,
    SpyALIFLayerLPF,
    LIFLayerLPF,
    ALIFLayerLPF,
)

from .wilson_cowan import (
    WilsonCowanLayer,
    WilsonCowanCURBDLayer,
)

from .leaky_integrate import (
    LILayer,
    SpyLILayer,
)


class LayerType(enum.Enum):
    LIF = 0
    ALIF = 1
    Izhikevich = 2
    LI = 3
    SpyLIF = 4
    SpyLI = 5
    SpyALIF = 6

    @classmethod
    def from_str(cls, name: str) -> Optional['LayerType']:
        """
        Get the LayerType from a string.

        :param name: The name of the LayerType.
        :type name: str

        :return: The LayerType.
        :rtype: Optional[LayerType]
        """
        if isinstance(name, LayerType):
            return name
        if name.startswith(cls.__name__):
            name = name.removeprefix(f"{cls.__name__}.")
        if name not in cls.__members__:
            return None
        return cls[name]


LayerType2Layer = {
    LayerType.LIF: LIFLayer,
    LayerType.ALIF: ALIFLayer,
    LayerType.Izhikevich: IzhikevichLayer,
    LayerType.LI: LILayer,
    LayerType.SpyLIF: SpyLIFLayer,
    LayerType.SpyALIF: SpyALIFLayer,
    LayerType.SpyLI: SpyLILayer,
}
