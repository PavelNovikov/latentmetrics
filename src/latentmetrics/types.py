from enum import Enum
from dataclasses import dataclass


class EstimateMethod(Enum):
    VALUE = "value"
    RANK = "rank"


class VariableType(Enum):
    BINARY = "binary"
    ORDINAL = "ordinal"
    CONTINUOUS = "continuous"


@dataclass
class CorrResult:
    estimate: float
    method: EstimateMethod
    x_type: VariableType
    y_type: VariableType
    resolved_name: str
