"""Local Transformation base for the synthetic_data degradation pipeline.

Mirrors the small subset of ``uni2ts.transform._base`` that
``degradation.py`` and the pretraining sources actually use, so this
package is self-contained and does not require importing all of uni2ts at
data-loading time.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


class Transformation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]: ...

    def chain(self, other: "Transformation") -> "Chain":
        return Chain([self, other])

    def __add__(self, other: "Transformation") -> "Chain":
        return self.chain(other)

    def __radd__(self, other):
        if other == 0:
            return self
        return other + self


@dataclass
class Chain(Transformation):
    """Compose a list of Transformations sequentially."""

    transformations: list[Transformation]

    def __post_init__(self) -> None:
        flat: list[Transformation] = []
        for t in self.transformations:
            if isinstance(t, Identity):
                continue
            if isinstance(t, Chain):
                flat.extend(t.transformations)
            else:
                assert isinstance(t, Transformation)
                flat.append(t)
        self.transformations = flat

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        for t in self.transformations:
            data_entry = t(data_entry)
        return data_entry


class Identity(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        return data_entry
