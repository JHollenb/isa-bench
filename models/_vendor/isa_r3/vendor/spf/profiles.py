"""SPF runtime profile resolver.

A profile names the regime a caller is operating in. The vocabulary
is closed; unknown names fail loudly. The default is always
``reference`` — the canonical Python/C path the spec test suites
grade against.

See ``docs/research/default-promotion-policy.md`` for the policy that
governs *what* each profile is allowed to substitute. This module is
the plumbing — it answers "which profile is active?" — and does not
itself flip any kernel.

Resolution precedence (highest first):

1. Explicit ``profile`` argument to ``resolve_profile``.
2. ``SPF_PROFILE`` environment variable.
3. The default ``reference``.

Example::

    from spf.profiles import resolve_profile

    profile = resolve_profile()                 # honours SPF_PROFILE
    profile = resolve_profile("hardware_pl")    # explicit override
    profile = resolve_profile(None)             # same as no argument

Unknown names raise ``ValueError`` with the allowed set.
"""

from __future__ import annotations

import os
from typing import Final, Literal

Profile = Literal[
    "reference",
    "hardware_pl",
    "embedded_rom",
    "sparse_dsp",
    "experimental_ml",
]

PROFILES: Final[frozenset[str]] = frozenset(
    {"reference", "hardware_pl", "embedded_rom", "sparse_dsp", "experimental_ml"}
)

DEFAULT_PROFILE: Final[Profile] = "reference"

PROFILE_ENV_VAR: Final[str] = "SPF_PROFILE"


def resolve_profile(
    profile: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Profile:
    """Pick the active profile under the documented precedence.

    Args:
        profile: Explicit caller override. Wins over environment.
            ``None`` defers to the environment (and then the default).
        env: Environment mapping to consult; defaults to
            ``os.environ``. Pulled out so unit tests can probe the
            resolver without mutating real env vars.

    Returns:
        The resolved profile name. Always one of ``PROFILES``.

    Raises:
        ValueError: If either ``profile`` or the environment value is
            non-empty and not in ``PROFILES``. Empty / missing values
            fall through to the next precedence level rather than
            raising.
    """
    if profile is not None:
        return _validate(profile, source="profile argument")

    if env is None:
        env = dict(os.environ)
    raw = env.get(PROFILE_ENV_VAR)
    if raw:
        return _validate(raw, source=f"${PROFILE_ENV_VAR}")

    return DEFAULT_PROFILE


def _validate(value: str, *, source: str) -> Profile:
    if value not in PROFILES:
        raise ValueError(
            f"unknown SPF profile {value!r} from {source}; "
            f"allowed: {sorted(PROFILES)}"
        )
    return value  # type: ignore[return-value]
