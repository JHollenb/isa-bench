"""Local import compatibility for the experiment runtime package."""
from __future__ import annotations

import enum
import typing
from pathlib import Path


# phase-accumulator-network-research main resolves the Bob/SPF tensor package
# through the sibling spectral-phase-format-research checkout.
DEFAULT_BOB_SPF_SRC = Path("/Users/jakeholl/domains/spectral-phase-format-research/src")


def ensure_scripts_importable() -> None:
    """Compatibility no-op retained for older runtime call sites."""


def install_python310_shims() -> None:
    """Install tiny shims needed by Bob SPF when this repo runs on Python 3.10."""

    if not hasattr(enum, "StrEnum"):
        class StrEnum(str, enum.Enum):
            def __str__(self) -> str:
                return str(self.value)

        enum.StrEnum = StrEnum  # type: ignore[attr-defined]
    if not hasattr(typing, "assert_never"):
        def assert_never(value: object) -> typing.NoReturn:
            raise AssertionError(f"Expected code to be unreachable, got: {value!r}")

        typing.assert_never = assert_never  # type: ignore[attr-defined]


ensure_scripts_importable()
