"""Serialise episodes into sharded records.

open-dreamer reads training data through ``grain.sources.ArrayRecordDataSource``
over files named ``shard-NNNNN.array_record``. This module writes that layout
when the ``array_record`` package is installed, and falls back to ``.npz``
shards otherwise so that collection, tests, and inspection all work on a
machine without the training stack.

The record schema mirrors the field names open-dreamer's own datasets use
(``video``, ``video_shape``, ``actions``) and adds ``poses``, which is specific
to od-MPC and ignored by the upstream loader.

The continuous action channel key (A1) has been confirmed against open-dreamer
source: its ``Actions.from_dict`` reads ``actions["continuous"]`` and requires
all three channel keys (``binary``, ``categorical``, ``continuous``) to be
present, raising ``KeyError`` on any missing key. The ``actions`` sub-dict is
therefore written with all three keys, the two unused channels set to ``None``,
so it is directly consumable by ``Actions.from_dict`` without a ``KeyError``.
See docs/SPEC.md §2.2 and §7.

Plain ``msgpack.packb`` has no native encoding for numpy arrays and raises
``TypeError`` on one (A1b); the ``array_record`` write path passes msgpack's
own ``default`` hook (``_msgpack_default``) to encode arrays as a tagged dict
instead.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import numpy as np

from .collector import Episode

__all__ = ["ShardWriter", "episode_to_record", "SHARD_STEM"]

SHARD_STEM = "shard"


def _msgpack_default(obj: object) -> dict[str, object]:
    """Encode a value msgpack has no native type for.

    Plain msgpack can only serialise its built-in types and raises
    ``TypeError`` on anything else (a numpy array, here); this is msgpack's
    own ``default`` hook mechanism for extending it, not a scheme copied from
    any upstream project. Arrays round-trip as a tagged dict carrying raw
    bytes, shape, and dtype.

    Raises:
        TypeError: If ``obj`` is not a numpy array.
    """
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not msgpack-serialisable")


def episode_to_record(episode: Episode) -> dict[str, object]:
    """Convert an episode into a serialisable record.

    Frames are kept as raw uint8 bytes plus an explicit shape rather than a
    nested list, which keeps records compact and reconstruction unambiguous.

    Args:
        episode: The episode to convert.

    Returns:
        A dict of plain Python and numpy values.
    """
    return {
        "video": episode.frames.tobytes(),
        "video_shape": list(episode.frames.shape),
        # All three channel keys are present so the dict round-trips through
        # open-dreamer's Actions.from_dict, which indexes every key; see the
        # module docstring and docs/SPEC.md §2.2.
        "actions": {
            "binary": None,
            "categorical": None,
            "continuous": episode.actions.astype(np.float32),
        },
        "poses": episode.poses.astype(np.float32),
    }


def _record_to_npz_arrays(record: dict[str, object], index: int) -> dict[str, np.ndarray]:
    """Flatten a record into arrays for the ``.npz`` fallback."""
    actions = record["actions"]
    assert isinstance(actions, dict)
    return {
        f"video_{index}": np.frombuffer(record["video"], dtype=np.uint8),  # type: ignore[arg-type]
        f"video_shape_{index}": np.array(record["video_shape"], dtype=np.int64),
        f"actions_continuous_{index}": actions["continuous"],
        f"poses_{index}": record["poses"],  # type: ignore[dict-item]
    }


class ShardWriter:
    """Write episodes into rotating shards.

    Use as a context manager so the final shard is always closed::

        with ShardWriter(out_dir, episodes_per_shard=50) as writer:
            for episode in episodes:
                writer.write(episode)

    Args:
        output_dir: Directory to create shards in.
        episodes_per_shard: Episodes before rotating to a new shard.
        prefer_array_record: Use ArrayRecord when the package is importable.
    """

    def __init__(
        self,
        output_dir: Path | str,
        episodes_per_shard: int = 50,
        prefer_array_record: bool = True,
    ) -> None:
        if episodes_per_shard < 1:
            raise ValueError(
                f"episodes_per_shard must be at least 1, got {episodes_per_shard}."
            )
        self.output_dir = Path(output_dir)
        self.episodes_per_shard = episodes_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._backend = "npz"
        self._array_record_writer_cls = None
        if prefer_array_record:
            self._try_enable_array_record()

        self._writer = None
        self._pending: list[dict[str, np.ndarray]] = []
        self._shard_index = 0
        self._in_shard = 0
        self._total = 0

    def _try_enable_array_record(self) -> None:
        try:
            from array_record.python.array_record_module import (  # type: ignore[import-not-found]
                ArrayRecordWriter,
            )
        except ImportError:
            return
        try:
            import msgpack  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            return
        self._array_record_writer_cls = ArrayRecordWriter
        self._backend = "array_record"

    @property
    def backend(self) -> str:
        """Either ``"array_record"`` or ``"npz"``."""
        return self._backend

    @property
    def total_episodes(self) -> int:
        """Episodes written so far."""
        return self._total

    @property
    def num_shards(self) -> int:
        """Shards opened so far."""
        return self._shard_index

    def _shard_path(self, suffix: str) -> Path:
        return self.output_dir / f"{SHARD_STEM}-{self._shard_index:05d}{suffix}"

    def _rotate(self) -> None:
        self._flush()
        if self._backend == "array_record":
            path = self._shard_path(".array_record")
            assert self._array_record_writer_cls is not None
            self._writer = self._array_record_writer_cls(str(path), "group_size:1")
        self._shard_index += 1
        self._in_shard = 0

    def write(self, episode: Episode) -> None:
        """Append one episode, rotating shards as needed."""
        if self._in_shard >= self.episodes_per_shard or (
            self._shard_index == 0 and self._in_shard == 0
        ):
            self._rotate()

        record = episode_to_record(episode)

        if self._backend == "array_record":
            import msgpack  # type: ignore[import-not-found]

            assert self._writer is not None
            self._writer.write(
                msgpack.packb(record, use_bin_type=True, default=_msgpack_default)
            )
        else:
            self._pending.append(_record_to_npz_arrays(record, self._in_shard))

        self._in_shard += 1
        self._total += 1

    def _flush(self) -> None:
        """Close the active shard, writing buffered records for the npz backend."""
        if self._backend == "array_record":
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            return

        if not self._pending:
            return
        merged: dict[str, np.ndarray] = {}
        for entry in self._pending:
            merged.update(entry)
        # _shard_index has already advanced past the shard being flushed.
        path = self.output_dir / f"{SHARD_STEM}-{self._shard_index - 1:05d}.npz"
        # Written via the positional-args form: record keys are data, and
        # **merged could otherwise collide with savez's own keyword params.
        with path.open("wb") as handle:
            np.savez_compressed(handle, **merged)  # type: ignore[arg-type]
        self._pending = []

    def close(self) -> None:
        """Close the writer and finalise the last shard."""
        self._flush()

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
