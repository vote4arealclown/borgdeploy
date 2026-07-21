"""Tests for the RTSP camera client and manager."""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from borg.modules.camera import CameraClient, CameraManager, _camera_credentials


class _FakeCapture:
    def __init__(self, frames: list[Any] | None = None) -> None:
        self._frames = frames or []
        self._index = 0
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, Any]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self._opened = False


def test_camera_builds_rtsp_url() -> None:
    cam = CameraClient(
        name="test",
        host="10.0.0.91",
        username="admin",
        password="secret",
        rtsp_port=554,
        rtsp_path="/stream1",
    )
    assert cam._build_rtsp_url() == "rtsp://admin:secret@10.0.0.91:554/stream1"
    assert cam.rtsp_url == "rtsp://10.0.0.91:554/stream1"


def test_camera_status() -> None:
    cam = CameraClient(name="test", host="10.0.0.91")
    status = cam.status()
    assert status["name"] == "test"
    assert status["rtsp_url"] == "rtsp://10.0.0.91:554/stream1"


def test_camera_available_reflects_opencv(monkeypatch) -> None:
    monkeypatch.setattr("borg.modules.camera.OPENCV_AVAILABLE", False)
    cam = CameraClient(name="test", host="10.0.0.91")
    assert cam.is_available is False


def test_camera_start_no_opencv(monkeypatch) -> None:
    monkeypatch.setattr("borg.modules.camera.OPENCV_AVAILABLE", False)
    cam = CameraClient(name="test", host="10.0.0.91")
    cam.start()
    assert cam._running is False
    assert "OpenCV is not installed" in cam._last_error


def test_camera_capture_loop(monkeypatch) -> None:
    import numpy as np

    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_capture = _FakeCapture([fake_frame])

    monkeypatch.setattr("borg.modules.camera.OPENCV_AVAILABLE", True)
    monkeypatch.setattr("borg.modules.camera.cv2.VideoCapture", lambda url: fake_capture)
    monkeypatch.setattr("borg.modules.camera.cv2.imencode", lambda ext, frame, params=None: (True, np.frombuffer(b"fakejpeg", dtype=np.uint8)))

    cam = CameraClient(name="test", host="10.0.0.91", username="admin", password="secret")
    cam.reconnect_seconds = 0.01
    cam.start()
    time.sleep(0.1)
    cam.stop()

    frame_bytes = cam.get_frame_bytes()
    assert frame_bytes == b"fakejpeg"


def test_camera_get_frame_bytes_without_frame(monkeypatch) -> None:
    monkeypatch.setattr("borg.modules.camera.OPENCV_AVAILABLE", True)
    cam = CameraClient(name="test", host="10.0.0.91")
    assert cam.get_frame_bytes() is None


def test_camera_mjpeg_stream_yields_chunks(monkeypatch) -> None:
    import numpy as np

    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    fake_capture = _FakeCapture([fake_frame, fake_frame])

    monkeypatch.setattr("borg.modules.camera.OPENCV_AVAILABLE", True)
    monkeypatch.setattr("borg.modules.camera.cv2.VideoCapture", lambda url: fake_capture)
    monkeypatch.setattr("borg.modules.camera.cv2.imencode", lambda ext, frame, params=None: (True, np.frombuffer(b"fakejpeg", dtype=np.uint8)))

    cam = CameraClient(name="test", host="10.0.0.91")
    cam.reconnect_seconds = 0.01
    cam.start()
    time.sleep(0.05)

    gen = cam.mjpeg_stream()
    chunk = next(gen)
    assert b"--frame" in chunk
    assert b"Content-Type: image/jpeg" in chunk
    assert b"fakejpeg" in chunk
    cam.stop()


def test_camera_credentials_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("BORG_CAMERA_FRONT_USERNAME", "user1")
    monkeypatch.setenv("BORG_CAMERA_FRONT_PASSWORD", "pass1")
    user, pwd = _camera_credentials("front", 1)
    assert user == "user1"
    assert pwd == "pass1"


def test_camera_credentials_fallback_by_index(monkeypatch) -> None:
    from borg.modules import camera as camera_module

    monkeypatch.setattr(camera_module, "_dotenv_values", {})
    monkeypatch.delenv("BORG_CAMERA_BACK_USERNAME", raising=False)
    monkeypatch.delenv("BORG_CAMERA_BACK_PASSWORD", raising=False)
    monkeypatch.setenv("BORG_CAMERA_2_USERNAME", "user2")
    monkeypatch.setenv("BORG_CAMERA_2_PASSWORD", "pass2")
    user, pwd = _camera_credentials("back", 2)
    assert user == "user2"
    assert pwd == "pass2"


def test_camera_manager_builds_clients(monkeypatch) -> None:
    monkeypatch.setattr(
        "borg.config.settings.cameras",
        [
            {"name": "front", "enabled": True, "host": "10.0.0.91", "rtsp_path": "/stream1"},
            {"name": "back", "enabled": True, "host": "10.0.0.42", "rtsp_path": "/stream2"},
            {"name": "disabled", "enabled": False, "host": "10.0.0.43"},
        ],
    )
    monkeypatch.setenv("BORG_CAMERA_FRONT_USERNAME", "admin")
    monkeypatch.setenv("BORG_CAMERA_FRONT_PASSWORD", "secret1")
    monkeypatch.setenv("BORG_CAMERA_BACK_USERNAME", "admin")
    monkeypatch.setenv("BORG_CAMERA_BACK_PASSWORD", "secret2")

    manager = CameraManager()
    assert sorted(manager.client_names) == ["back", "front"]
    front = manager.get_client("front")
    assert front is not None
    assert front.host == "10.0.0.91"
    assert front.password == "secret1"
    back = manager.get_client("back")
    assert back is not None
    assert back.host == "10.0.0.42"


def test_camera_manager_status(monkeypatch) -> None:
    monkeypatch.setattr(
        "borg.config.settings.cameras",
        [{"name": "front", "enabled": True, "host": "10.0.0.91"}],
    )
    manager = CameraManager()
    statuses = manager.status()
    assert len(statuses) == 1
    assert statuses[0]["name"] == "front"
    single = manager.status("front")
    assert single["name"] == "front"
    missing = manager.status("missing")
    assert "error" in missing
