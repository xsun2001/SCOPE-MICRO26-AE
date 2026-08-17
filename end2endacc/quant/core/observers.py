"""Small calibration observers required by the Figure 16 source snapshot."""

from __future__ import annotations

import torch


class VectorAbsMaxObserver:
    """Accumulate an elementwise absolute maximum on CPU."""

    def __init__(self) -> None:
        self._value: torch.Tensor | None = None

    def update(self, value: torch.Tensor) -> None:
        current = value.detach().float().abs().cpu()
        if self._value is None:
            self._value = current.clone()
            return
        if self._value.shape != current.shape:
            raise ValueError(
                "VectorAbsMaxObserver shape changed from "
                f"{tuple(self._value.shape)} to {tuple(current.shape)}."
            )
        self._value = torch.maximum(self._value, current)

    @property
    def value(self) -> torch.Tensor:
        if self._value is None:
            raise ValueError("VectorAbsMaxObserver has not observed any values.")
        return self._value

    def to_list(self) -> list[float]:
        return [float(value) for value in self.value.reshape(-1).tolist()]
