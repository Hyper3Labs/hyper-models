"""The reported version must be the version that was packaged.

`__version__` was a hand-maintained literal and stayed at 0.3.0 through the
0.3.1 release. The demo Dockerfiles print it to confirm which catalog they
installed, so a stale value makes a wrong install look correct.
"""

from __future__ import annotations

from pathlib import Path

import hyper_models

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - tomllib is stdlib from 3.11
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]


def _declared_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(pyproject["project"]["version"])


def test_reported_version_matches_pyproject() -> None:
    assert hyper_models.__version__ == _declared_version()


def test_version_is_not_a_hand_maintained_literal() -> None:
    source = (ROOT / "src" / "hyper_models" / "__init__.py").read_text(encoding="utf-8")

    assert "importlib.metadata" in source, (
        "__version__ must be read from the installed distribution"
    )
