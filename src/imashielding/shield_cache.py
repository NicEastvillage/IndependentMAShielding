"""Persistent on-disk caching of synthesized shields.

Synthesis can take seconds to hours as the abstract state space grows, while the
result is a pure function of the transition system and the safety property it was
computed for.  This module stores that result on disk so later runs can load it
instead of synthesizing it again.

Format
------
A shield is written as a versioned, gzip-compressed pickle (``*.shield.pkl.gz``).
Pickle is the appropriate representation for the abstract states themselves --
in this project frozen dataclasses such as
:class:`imashielding.roads.RoadNetworkState` -- which a text/JSON format cannot
represent generically.  The expensive transient dependency graph is *not*
stored: only the set of states, their final colours, the release flags and the
precomputed safe actions are needed to answer shield queries.  The outer
envelope records the format name and version, so an incompatible or corrupt
file is rejected and the shield is recomputed rather than silently misread.

Only load cache files that you trust: unpickling can execute arbitrary code.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from imashielding.shield_graph import ShieldGraph

FORMAT_NAME = "imashielding.shield"
FORMAT_VERSION = 2

#: Environment variable that overrides the default cache directory.
CACHE_DIR_ENV = "IMASHIELDING_SHIELD_CACHE"

#: Compression level used when writing caches. The payload is dominated by
#: state objects and small integer arrays, so a medium level is a better
#: time/space trade-off than gzip's slow default of 9.
COMPRESSION_LEVEL = 3

PathLike = Union[str, os.PathLike]


class ShieldCacheError(RuntimeError):
    """A shield cache file is missing, corrupt, or written in an incompatible format."""


def default_cache_dir() -> Path:
    """The directory used when no explicit cache directory is given.

    Override with the ``IMASHIELDING_SHIELD_CACHE`` environment variable.  The
    default lives next to the ``IndependentMAShielding`` project so that shields
    are shared between the library demos and the BenchMARL task, no matter which
    directory the process is started from.
    """
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    # <repo>/IndependentMAShielding/src/imashielding/shield_cache.py -> <repo>/IndependentMAShielding
    return Path(__file__).resolve().parents[2] / "shield_cache"


def resolve_cache_dir(directory: Optional[PathLike] = None) -> Path:
    if directory is None:
        return default_cache_dir()
    return Path(directory).expanduser()


def _readable_slug(cache_key: str, max_length: int = 64) -> str:
    """Turn a cache key into a human-recognisable, filesystem-safe prefix."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", cache_key).strip("-._")
    slug = slug[:max_length].rstrip("-._")
    return slug or "shield"


def shield_cache_path(cache_key: str, cache_dir: Optional[PathLike] = None) -> Path:
    """Return the deterministic cache file for ``cache_key``.

    The digest guarantees uniqueness; the readable slug makes it obvious which
    configuration a file belongs to when browsing the cache directory.
    """
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    return resolve_cache_dir(cache_dir) / f"{_readable_slug(cache_key)}-{digest}.shield.pkl.gz"


def save_shield(
    graph: ShieldGraph,
    path: PathLike,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Atomically write ``graph`` and its metadata to ``path``.

    The file is first written to a temporary sibling and then moved into place, so
    a crashed or concurrent writer can never leave a half-written shield behind.
    """
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "created_at": time.time(),
        "metadata": dict(metadata or {}),
        "states": list(graph.states),
        "colors": bytes(graph.colors),
        "released": bytes(graph.released),
        "safe_actions": graph.safe_actions,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as raw, gzip.GzipFile(
            fileobj=raw, mode="wb", compresslevel=COMPRESSION_LEVEL, mtime=0
        ) as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_shield(path: PathLike) -> Tuple[ShieldGraph, Dict[str, Any]]:
    """Read a shield and its metadata, raising :class:`ShieldCacheError` on failure."""
    path = Path(path)
    try:
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
    except Exception as error:
        # A truncated file, an incompatible pickle, a moved class, a gzip
        # error, ... all mean the same thing to the caller: this cache entry is
        # unusable.
        raise ShieldCacheError(f"Could not read shield cache {path}: {error}") from error

    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ShieldCacheError(f"{path} is not an {FORMAT_NAME} file")

    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ShieldCacheError(
            f"{path} uses shield format version {version!r}; expected {FORMAT_VERSION}"
        )

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ShieldCacheError(f"{path} contains invalid metadata")

    try:
        states = payload["states"]
        colors = payload["colors"]
        released = payload["released"]
        safe_actions = payload["safe_actions"]
        graph = ShieldGraph(states, colors, released, safe_actions)
    except (KeyError, TypeError, ValueError) as error:
        raise ShieldCacheError(f"{path} contains an invalid compact shield: {error}") from error

    return graph, metadata
