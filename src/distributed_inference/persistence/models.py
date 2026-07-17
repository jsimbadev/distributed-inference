"""Model persistence references based on importable Python callables."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from distributed_inference.errors import ManifestError, ModelError
from distributed_inference.model import Model


class ModelBuilder(Protocol):
    """Importable callable that builds a runtime model from config."""

    __module__: str
    __qualname__: str

    def __call__(self, config: Mapping[str, Any]) -> Model: ...


@dataclass(frozen=True)
class PythonCallableSpec:
    """Serializable reference to an importable Python callable."""

    module: str
    qualname: str
    project_root: str
    source_path: str

    @classmethod
    def from_callable(
        cls,
        builder: ModelBuilder,
        *,
        project_root: Path,
    ) -> PythonCallableSpec:
        """Capture durable import metadata for a model builder."""
        root = project_root.resolve()
        if "<locals>" in builder.__qualname__:
            msg = (
                f"Cannot persist local callable {builder.__qualname__!r}; "
                "model builders must be importable from module scope."
            )
            raise ValueError(msg)
        source = inspect.getsourcefile(builder)
        if source is None:
            msg = f"Cannot determine source file for {builder!r}."
            raise ValueError(msg)
        source_path = Path(source).resolve().relative_to(root)
        return cls(
            module=builder.__module__,
            qualname=builder.__qualname__,
            project_root=str(root),
            source_path=source_path.as_posix(),
        )

    def to_manifest(self) -> dict[str, str]:
        """Return a serializable callable reference."""
        return {
            "module": self.module,
            "qualname": self.qualname,
            "project_root": self.project_root,
            "source_path": self.source_path,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> PythonCallableSpec:
        """Rehydrate a callable reference from a manifest payload."""
        return cls(
            module=str(payload["module"]),
            qualname=str(payload["qualname"]),
            project_root=str(payload["project_root"]),
            source_path=str(payload["source_path"]),
        )

    def resolve(self) -> ModelBuilder:
        """Resolve the callable reference to a runtime builder."""
        project_root = Path(self.project_root).resolve()
        source_path = (project_root / self.source_path).resolve()
        if not source_path.is_file():
            msg = f"Model source file does not exist: {source_path}."
            raise ModelError(msg)

        module = _load_module(self.module, source_path, project_root=project_root)
        builder = _resolve_qualname(module, self.qualname)
        if not callable(builder):
            msg = f"Resolved object {self.qualname!r} is not callable."
            raise ModelError(msg)
        return cast(ModelBuilder, builder)


@dataclass(frozen=True)
class ModelSpec:
    """Serializable model definition using an importable builder callable."""

    callable: PythonCallableSpec
    version: str
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_callable(
        cls,
        builder: ModelBuilder,
        *,
        config: Mapping[str, Any],
        project_root: Path,
        version: str,
    ) -> ModelSpec:
        """Create a serializable model spec from a local builder callable."""
        return cls(
            callable=PythonCallableSpec.from_callable(
                builder,
                project_root=project_root,
            ),
            version=version,
            config=dict(config),
        )

    def to_model(self) -> Model:
        """Rehydrate the model builder and construct a runtime model."""
        builder = self.callable.resolve()
        model = builder(dict(self.config))
        if not _is_model_compatible(model):
            msg = (
                f"Model builder {self.callable.qualname!r} did not return a "
                "callable model with model metadata."
            )
            raise ModelError(msg)
        return model

    def to_manifest(self) -> dict[str, Any]:
        """Return a serializable model manifest."""
        return {
            "kind": "python-callable",
            "version": self.version,
            "callable": self.callable.to_manifest(),
            "config": dict(self.config),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> ModelSpec:
        """Rehydrate a model spec from a manifest payload."""
        if payload["kind"] != "python-callable":
            msg = f"Unsupported model manifest kind {payload['kind']!r}."
            raise ManifestError(msg)
        callable_payload = payload["callable"]
        if not isinstance(callable_payload, Mapping):
            msg = "Model manifest callable reference must be a mapping."
            raise ManifestError(msg)
        config = payload.get("config", {})
        if not isinstance(config, Mapping):
            msg = "Model manifest config must be a mapping."
            raise ManifestError(msg)
        return cls(
            callable=PythonCallableSpec.from_manifest(callable_payload),
            version=str(payload["version"]),
            config=dict(config),
        )


def _load_module(
    module_name: str, source_path: Path, *, project_root: Path
) -> ModuleType:
    loaded = sys.modules.get(module_name)
    if loaded is not None and _module_file(loaded) == source_path:
        return loaded

    project_root_path = str(project_root)
    remove_project_root = project_root_path not in sys.path
    if remove_project_root:
        sys.path.insert(0, project_root_path)

    try:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            module = _load_module_from_source(module_name, source_path)
    finally:
        if remove_project_root:
            sys.path.remove(project_root_path)

    if _module_file(module) != source_path:
        module = _load_module_from_source(module_name, source_path)
    return module


def _load_module_from_source(module_name: str, source_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        msg = f"Could not import model module {module_name!r} from {source_path}."
        raise ModelError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        msg = f"Could not import model module {module_name!r} from {source_path}."
        raise ModelError(msg) from exc
    return module


def _module_file(module: ModuleType) -> Path | None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    return Path(module_file).resolve()


def _resolve_qualname(module: ModuleType, qualname: str) -> object:
    current: object = module
    for part in qualname.split("."):
        if part == "<locals>" or not hasattr(current, part):
            msg = f"Could not resolve {qualname!r} in module {module.__name__!r}."
            raise ModelError(msg)
        current = getattr(current, part)
    return current


def _is_model_compatible(candidate: object) -> bool:
    return callable(candidate) and hasattr(candidate, "info")
