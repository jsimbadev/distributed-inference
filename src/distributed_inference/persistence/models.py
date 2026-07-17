"""Model persistence references based on importable Python callables."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
        raise NotImplementedError

    def to_manifest(self) -> dict[str, Any]:
        """Return a serializable model manifest."""
        return {
            "kind": "python-callable",
            "version": self.version,
            "callable": self.callable.to_manifest(),
            "config": dict(self.config),
        }
