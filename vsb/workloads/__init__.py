from enum import Enum, unique
from .base import VectorWorkload, VectorWorkloadSequence, SingleVectorWorkloadSequence


@unique
class Workload(Enum):
    """Set of supported workloads, the value is the string used to
    specify via --workload=
    """

    Mnist = "mnist"
    MnistTest = "mnist-test"
    Nq768 = "nq768"
    Nq768Test = "nq768-test"
    YFCC = "yfcc-10M"
    YFCCTest = "yfcc-test"
    Cohere768 = "cohere768"
    Cohere768Test = "cohere768-test"

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
            case Workload.YFCC:
                from .yfcc.yfcc import YFCC

                return YFCC
            case Workload.YFCCTest:
                from .yfcc.yfcc import YFCCTest

                return YFCCTest

            case Workload.Cohere768:
                from .cohere_768.cohere_768 import Cohere768

                return Cohere768

            case Workload.Cohere768Test:
                from .cohere_768.cohere_768 import Cohere768Test

                return Cohere768Test

    def describe(self) -> tuple[str, int, int, str, int]:
        """Return a tuple with attributes of the workload: name, dataset size, dimensionality, distance metric, and query count."""
        cls = self._get_class()
        return (
            self.value,
            cls.record_count(),
            cls.dimensions(),
            cls.metric().value,
            cls.request_count(),
        )


@unique
class WorkloadSequence(Enum):
    """Set of supported workload sequences, the value is the string used to
    specify via --workload=.
    """

    MnistSplit = "mnist-split"
    Nq768Split = "nq768-split"

    def build(self, **kwargs) -> VectorWorkloadSequence:
        """Construct an instance of VectorWorkload based on the value of the enum."""
        cls = self._get_class()
        return cls(self.value, **kwargs)

    def _get_class(self) -> type[VectorWorkloadSequence]:
        """Return the VectorWorkloadSequence class to use, based on the value of the enum"""
        match self:
            case WorkloadSequence.MnistSplit:
                from .mnist.mnist import MnistSplit

                return MnistSplit
            case WorkloadSequence.Nq768Split:
                from .nq_768_tasb.nq_768_tasb import Nq768TasbSplit

                return Nq768TasbSplit
        pass


def build_workload_sequence(name: str, **kwargs) -> VectorWorkloadSequence:
    """Takes either a Workload or WorkloadSequence name and returns the corresponding
    WorkloadSequence. Workloads will be wrapped into single-element WorkloadSequences.
    """
    try:
        return WorkloadSequence(name).build(**kwargs)
    except ValueError:
        # Try to build a Workload.
        workload = Workload(name).build(**kwargs)
        return SingleVectorWorkloadSequence(name, workload)
