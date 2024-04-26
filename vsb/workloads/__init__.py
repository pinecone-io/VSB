from enum import Enum
from .base import VectorWorkload


class Workload(Enum):
    """Set of supported workloads, the value is the string used to
    specify via --benchmark=
    """

    MNIST = "mnist"

    def build(self) -> VectorWorkload:
        """Construct an instance of Benchmark based on the value of the enum"""
        match self:
            case Workload.MNIST:
                from .mnist.mnist import MNIST
                return MNIST()
