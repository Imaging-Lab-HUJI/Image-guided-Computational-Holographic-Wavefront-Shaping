from enum import Enum


class QualityMetric(Enum):
    """
    This class is an enumeration of the available image quality metrics.
    """
    VARIANCE = 1
    ENTROPY = 2
    FOURIER_VARIANCE = 3


class PhaseInit(Enum):
    ZEROS = 0
    RANDOM = 1


class FieldInit(Enum):
    ALL = 0
    RANDOM = 1
    LINEAR = 2
    MIN_CORRELATION = 3
