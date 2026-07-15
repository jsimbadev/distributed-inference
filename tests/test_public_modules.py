from distributed_inference import Bounds, CallableModel, ModelError, TransformedModel
from distributed_inference.bounds import Bounds as BoundsFromModule
from distributed_inference.errors import ModelError as ModelErrorFromModule
from distributed_inference.model import CallableModel as CallableModelFromModule
from distributed_inference.transforms import (
    TransformedModel as TransformedModelFromModule,
)


def test_bounds_export_matches_module() -> None:
    assert Bounds is BoundsFromModule


def test_model_export_matches_module() -> None:
    assert CallableModel is CallableModelFromModule


def test_error_export_matches_module() -> None:
    assert ModelError is ModelErrorFromModule


def test_transform_export_matches_module() -> None:
    assert TransformedModel is TransformedModelFromModule
