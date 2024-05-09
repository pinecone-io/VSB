from enum import Enum, unique
from .base import VectorWorkload


@unique
class Workload(Enum):
    """Set of supported workloads, the value is the string used to
    specify via --benchmark=
    """

    Mnist = "mnist"
    MnistTest = "mnist-test"
    Nq768 = "nq768"
    Nq768Test = "nq768-test"

    def build(self, **kwargs) -> VectorWorkload:
        """Construct an instance of VectorWorkload based on the value of the enum."""
        cls = self._get_class()
        return cls(self.value, **kwargs)

    def _get_class(self) -> type[VectorWorkload]:
        """Return the VectorWorkload class to use, based on the value of the enum"""
        match self:
            case Workload.Mnist:
                from .mnist.mnist import Mnist

                return Mnist
            case Workload.MnistTest:
                from .mnist.mnist import MnistTest

                return MnistTest
            case Workload.Nq768:
                from .nq_768_tasb.nq_768_tasb import Nq768Tasb

                return Nq768Tasb
            case Workload.Nq768Test:
                from .nq_768_tasb.nq_768_tasb import Nq768TasbTest

                return Nq768TasbTest
