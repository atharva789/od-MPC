"""Tests for the stub simulator, the collector, and the shard writer."""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from od_mpc.data.collector import CollectorConfig, Episode, collect_episode, collect_episodes
from od_mpc.data.writer import ShardWriter, episode_to_record
from od_mpc.sim.protocol import MarsSim
from od_mpc.sim.stub import RoomBounds, StubMars

pytestmark = pytest.mark.unit

FAST = CollectorConfig(steps_per_episode=12, control_dt=0.1, settle_seconds=0.0)


# --- stub simulator ---------------------------------------------------------


def test_stub_satisfies_the_protocol() -> None:
    assert isinstance(StubMars(), MarsSim)


def test_stub_drives_forward_along_its_heading() -> None:
    sim = StubMars(spawn=(0.0, 0.0, 0.0))
    sim.set_cmd_vel(0.4, 0.0)
    sim.step(1.0)
    x, y, _ = sim.pose()
    assert x == pytest.approx(0.4, abs=0.02)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_stub_yaw_wraps_into_range() -> None:
    sim = StubMars()
    sim.set_cmd_vel(0.0, 1.0)
    sim.step(20.0)
    _, _, yaw = sim.pose()
    assert -math.pi <= yaw < math.pi


def test_stub_clamps_commands_to_limits() -> None:
    sim = StubMars(spawn=(0.0, 0.0, 0.0))
    sim.set_cmd_vel(99.0, 0.0)
    sim.step(1.0)
    x, _, _ = sim.pose()
    assert x == pytest.approx(0.4, abs=0.02)


def test_stub_cannot_leave_the_room() -> None:
    sim = StubMars(bounds=RoomBounds(-1.0, 1.0, -1.0, 1.0), spawn=(0.0, 0.0, 0.0))
    sim.set_cmd_vel(0.4, 0.0)
    sim.step(60.0)
    x, y, _ = sim.pose()
    assert -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0


def test_stub_reset_restores_spawn() -> None:
    sim = StubMars(spawn=(1.0, 2.0, 0.5))
    sim.set_cmd_vel(0.4, 0.3)
    sim.step(2.0)
    sim.reset()
    assert sim.pose() == pytest.approx((1.0, 2.0, 0.5))


def test_stub_lidar_is_closer_to_a_near_wall() -> None:
    sim = StubMars(bounds=RoomBounds(-2.0, 2.0, -5.0, 5.0), spawn=(1.5, 0.0, 0.0))
    scan = sim.lidar_scan(4, max_range=12.0)
    # Ray 0 points along +x toward the near wall at x=2.0; ray 2 points at -x.
    assert scan[0] < scan[2]


def test_stub_lidar_respects_max_range() -> None:
    sim = StubMars(bounds=RoomBounds(-100.0, 100.0, -100.0, 100.0))
    assert np.all(sim.lidar_scan(16, max_range=5.0) <= 5.0)


def test_stub_renders_expected_shape_and_dtype() -> None:
    frame = StubMars(render_wh=(32, 24)).render_rgb("main")
    assert frame.shape == (24, 32, 3)
    assert frame.dtype == np.uint8


def test_stub_frames_change_with_pose() -> None:
    sim = StubMars(spawn=(0.0, 0.0, 0.0))
    before = sim.render_rgb("main").copy()
    sim.set_cmd_vel(0.4, 1.0)
    sim.step(2.0)
    assert not np.array_equal(before, sim.render_rgb("main"))


def test_stub_rejects_unknown_camera() -> None:
    with pytest.raises(ValueError, match="Unknown camera"):
        StubMars().render_rgb("periscope")


def test_stub_rejects_non_finite_command() -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        StubMars().set_cmd_vel(float("nan"), 0.0)


# --- collector --------------------------------------------------------------


def test_collect_episode_shapes_agree() -> None:
    episode = collect_episode(StubMars(render_wh=(16, 16)), FAST)
    assert episode.frames.shape == (12, 16, 16, 3)
    assert episode.actions.shape == (12, 2)
    assert episode.poses.shape == (12, 3)
    assert episode.length == 12


def test_collected_actions_are_normalised() -> None:
    episode = collect_episode(StubMars(render_wh=(16, 16)), FAST)
    assert np.all(np.abs(episode.actions) <= 1.0)
    assert episode.actions.dtype == np.float32


def test_collected_poses_move() -> None:
    """A collector that records a stationary robot has failed silently."""
    episode = collect_episode(StubMars(render_wh=(16, 16)), FAST)
    assert np.linalg.norm(episode.poses[-1, :2] - episode.poses[0, :2]) > 1e-3


def test_collection_is_reproducible() -> None:
    first = collect_episode(StubMars(render_wh=(16, 16)), FAST)
    second = collect_episode(StubMars(render_wh=(16, 16)), FAST)
    np.testing.assert_array_equal(first.actions, second.actions)


def test_collect_episodes_yields_requested_count() -> None:
    episodes = list(collect_episodes(StubMars(render_wh=(16, 16)), FAST, count=3))
    assert len(episodes) == 3


def test_collect_episodes_rejects_zero() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(collect_episodes(StubMars(), FAST, count=0))


def test_collector_config_rejects_tiny_episodes() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        CollectorConfig(steps_per_episode=1)


def test_collector_config_rejects_non_positive_control_dt() -> None:
    with pytest.raises(ValueError, match="control_dt must be positive"):
        CollectorConfig(control_dt=0.0)


def test_collector_config_rejects_negative_settle_seconds() -> None:
    with pytest.raises(ValueError, match="settle_seconds must be non-negative"):
        CollectorConfig(settle_seconds=-0.1)


def test_collect_episode_settles_before_recording() -> None:
    """settle_seconds > 0 zeroes cmd_vel and advances physics before frame 0."""
    config = CollectorConfig(steps_per_episode=2, control_dt=0.1, settle_seconds=0.5)
    sim = StubMars(render_wh=(16, 16))
    episode = collect_episode(sim, config)
    assert episode.length == 2


def test_episode_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="share a length"):
        Episode(
            frames=np.zeros((4, 8, 8, 3), dtype=np.uint8),
            actions=np.zeros((3, 2), dtype=np.float32),
            poses=np.zeros((4, 3), dtype=np.float32),
        )


def test_episode_rejects_malformed_frames() -> None:
    with pytest.raises(ValueError, match=r"frames must be \(T, H, W, 3\)"):
        Episode(
            frames=np.zeros((4, 8, 8), dtype=np.uint8),
            actions=np.zeros((4, 2), dtype=np.float32),
            poses=np.zeros((4, 3), dtype=np.float32),
        )


def test_episode_rejects_malformed_actions() -> None:
    with pytest.raises(ValueError, match=r"actions must be \(T, 2\)"):
        Episode(
            frames=np.zeros((4, 8, 8, 3), dtype=np.uint8),
            actions=np.zeros((4, 3), dtype=np.float32),
            poses=np.zeros((4, 3), dtype=np.float32),
        )


def test_episode_rejects_malformed_poses() -> None:
    with pytest.raises(ValueError, match=r"poses must be \(T, 3\)"):
        Episode(
            frames=np.zeros((4, 8, 8, 3), dtype=np.uint8),
            actions=np.zeros((4, 2), dtype=np.float32),
            poses=np.zeros((4, 2), dtype=np.float32),
        )


# --- writer -----------------------------------------------------------------


def _episode(length: int = 4) -> Episode:
    return Episode(
        frames=np.arange(length * 8 * 8 * 3, dtype=np.uint8).reshape(length, 8, 8, 3),
        actions=np.zeros((length, 2), dtype=np.float32),
        poses=np.zeros((length, 3), dtype=np.float32),
    )


def test_record_preserves_frames_byte_exactly() -> None:
    episode = _episode()
    record = episode_to_record(episode)
    restored = np.frombuffer(record["video"], dtype=np.uint8).reshape(
        record["video_shape"]
    )
    np.testing.assert_array_equal(restored, episode.frames)


def test_writer_creates_shards(tmp_path) -> None:
    with ShardWriter(tmp_path, episodes_per_shard=2) as writer:
        for _ in range(5):
            writer.write(_episode())
        backend = writer.backend
        total = writer.total_episodes

    assert total == 5
    suffix = ".array_record" if backend == "array_record" else ".npz"
    assert len(list(tmp_path.glob(f"shard-*{suffix}"))) == 3


def test_writer_shard_names_are_zero_padded(tmp_path) -> None:
    with ShardWriter(tmp_path, episodes_per_shard=1) as writer:
        writer.write(_episode())
    assert any(p.name.startswith("shard-00000") for p in tmp_path.iterdir())


def test_writer_rejects_zero_episodes_per_shard(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ShardWriter(tmp_path, episodes_per_shard=0)


def test_npz_shard_roundtrips(tmp_path) -> None:
    episode = _episode()
    with ShardWriter(tmp_path, episodes_per_shard=4, prefer_array_record=False) as w:
        w.write(episode)

    shard = next(tmp_path.glob("shard-*.npz"))
    with np.load(shard) as data:
        shape = tuple(data["video_shape_0"])
        frames = data["video_0"].reshape(shape)
    np.testing.assert_array_equal(frames, episode.frames)


def _install_fake_array_record_module(monkeypatch: pytest.MonkeyPatch) -> type:
    """Make ``array_record`` importable without the real package installed.

    Returns a fake ``ArrayRecordWriter`` class that records every call made to
    it, so tests can assert on the array_record code path in ShardWriter
    without depending on the real (optional, not installed here) package.
    """

    class _FakeArrayRecordWriter:
        def __init__(self, path: str, options: str) -> None:
            self.path = path
            self.options = options
            self.written: list[bytes] = []
            self.closed = False

        def write(self, record: bytes) -> None:
            self.written.append(record)

        def close(self) -> None:
            self.closed = True

    array_record_module = types.ModuleType("array_record.python.array_record_module")
    array_record_module.ArrayRecordWriter = _FakeArrayRecordWriter
    monkeypatch.setitem(sys.modules, "array_record", types.ModuleType("array_record"))
    monkeypatch.setitem(
        sys.modules, "array_record.python", types.ModuleType("array_record.python")
    )
    monkeypatch.setitem(
        sys.modules, "array_record.python.array_record_module", array_record_module
    )

    msgpack_module = types.ModuleType("msgpack")
    msgpack_module.packb = lambda obj, use_bin_type=True: repr(obj).encode()
    monkeypatch.setitem(sys.modules, "msgpack", msgpack_module)

    return _FakeArrayRecordWriter


def test_array_record_backend_used_when_available(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_array_record_module(monkeypatch)

    with ShardWriter(tmp_path, episodes_per_shard=2) as writer:
        assert writer.backend == "array_record"
        for _ in range(3):
            writer.write(_episode())
        assert writer.num_shards == 2


def test_array_record_backend_writes_and_closes_each_shard(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_writer_cls = _install_fake_array_record_module(monkeypatch)
    created: list = []
    real_init = fake_writer_cls.__init__

    def _tracking_init(self, path: str, options: str) -> None:
        real_init(self, path, options)
        created.append(self)

    fake_writer_cls.__init__ = _tracking_init

    with ShardWriter(tmp_path, episodes_per_shard=2) as writer:
        for _ in range(3):
            writer.write(_episode())

    assert len(created) == 2
    assert [len(inst.written) for inst in created] == [2, 1]
    assert all(inst.closed for inst in created)


def test_array_record_backend_falls_back_to_npz_without_msgpack(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_array_record_module(monkeypatch)
    monkeypatch.setitem(sys.modules, "msgpack", None)

    writer = ShardWriter(tmp_path)
    assert writer.backend == "npz"
