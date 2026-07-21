"""RTSP camera client that serves live MJPEG streams and snapshots."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

from borg.config import settings

logger = logging.getLogger(__name__)

# OpenCV is an optional runtime dependency; gracefully degrade if missing.
try:
    import cv2
    import numpy as np

    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


class CameraClient:
    """Read a single RTSP feed in a background thread and expose the latest frame."""

    def __init__(
        self,
        name: str,
        host: str,
        username: str = "",
        password: Optional[str] = None,
        rtsp_port: int = 554,
        rtsp_path: str = "/stream1",
        reconnect_seconds: float = 5.0,
        stream_quality: int = 75,
    ) -> None:
        self.name = name
        self.host = host
        self.username = username
        self.password = password or ""
        self.rtsp_port = rtsp_port
        self.rtsp_path = rtsp_path
        self.reconnect_seconds = reconnect_seconds
        self.stream_quality = stream_quality

        self._rtsp_url = self._build_rtsp_url()
        self._latest_frame: Optional[np.ndarray] = None  # type: ignore[name-defined]
        self._lock = threading.Lock()
        self._connected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_error: str = ""

    def _build_rtsp_url(self) -> str:
        creds = ""
        if self.username:
            creds = self.username
            if self.password:
                creds = f"{creds}:{self.password}"
            creds = f"{creds}@"
        return f"rtsp://{creds}{self.host}:{self.rtsp_port}{self.rtsp_path}"

    @property
    def rtsp_url(self) -> str:
        """Return the RTSP URL with credentials redacted for logging."""
        return f"rtsp://{self.host}:{self.rtsp_port}{self.rtsp_path}"

    @property
    def is_available(self) -> bool:
        return OPENCV_AVAILABLE

    def start(self) -> None:
        """Start the background capture thread."""
        if not OPENCV_AVAILABLE:
            self._last_error = "OpenCV is not installed; camera streaming is unavailable"
            logger.error(self._last_error)
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera '%s' capture thread started for %s", self.name, self.rtsp_url)

    def stop(self) -> None:
        """Signal the capture thread to stop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _capture_loop(self) -> None:
        cap: Optional[Any] = None
        while self._running:
            if cap is None or not cap.isOpened():
                self._connected = False
                logger.info("Connecting to camera '%s' at %s", self.name, self.rtsp_url)
                cap = cv2.VideoCapture(self._rtsp_url)
                if not cap.isOpened():
                    self._last_error = f"Unable to open RTSP stream at {self.rtsp_url}"
                    logger.warning(self._last_error)
                    if cap:
                        cap.release()
                        cap = None
                    time.sleep(self.reconnect_seconds)
                    continue
                self._connected = True
                self._last_error = ""
                logger.info("Camera '%s' connected", self.name)

            ret, frame = cap.read()
            if not ret:
                self._connected = False
                self._last_error = "Lost camera connection; reconnecting"
                logger.warning(self._last_error)
                cap.release()
                cap = None
                time.sleep(self.reconnect_seconds)
                continue

            with self._lock:
                self._latest_frame = frame

        if cap:
            cap.release()
        self._connected = False
        logger.info("Camera '%s' capture thread stopped", self.name)

    def get_frame_bytes(self, quality: Optional[int] = None) -> Optional[bytes]:
        """Return the latest frame encoded as JPEG, or None if unavailable."""
        if not OPENCV_AVAILABLE:
            return None
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return None
        q = quality if quality is not None else self.stream_quality
        ret, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ret:
            return None
        return encoded.tobytes()

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": True,
            "available": self.is_available,
            "connected": self._connected,
            "rtsp_url": self.rtsp_url,
            "last_error": self._last_error,
        }

    def mjpeg_stream(self, boundary: str = "frame"):
        """Generator yielding multipart JPEG chunks for an MJPEG HTTP stream."""
        while True:
            frame_bytes = self.get_frame_bytes()
            if frame_bytes is None:
                time.sleep(0.05)
                continue
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )


def _camera_env_prefix(name: str) -> str:
    """Derive an env-var prefix from the camera name, e.g. 'front' -> 'BORG_CAMERA_FRONT'."""
    return f"BORG_CAMERA_{name.upper().replace(' ', '_').replace('-', '_')}"


# Pydantic-settings reads .env into Settings fields, not necessarily os.environ.
# Load the same file so camera credential env vars are available here.
_dotenv_values: dict[str, Optional[str]] = {}
try:
    from dotenv import dotenv_values

    _env_file = getattr(settings, "model_config", {}).get("env_file")
    if _env_file:
        _dotenv_values = dotenv_values(_env_file)
except Exception:
    pass


def _env_or_dotenv(key: str) -> Optional[str]:
    return os.environ.get(key, _dotenv_values.get(key))


def _camera_credentials(name: str, index: int) -> tuple[str, Optional[str]]:
    """Return (username, password) for a camera from env vars / .env file.

    Looks up by name first (BORG_CAMERA_FRONT_USERNAME), then by index
    (BORG_CAMERA_1_USERNAME), then falls back to empty credentials.
    """
    prefix = _camera_env_prefix(name)
    username = _env_or_dotenv(f"{prefix}_USERNAME")
    password = _env_or_dotenv(f"{prefix}_PASSWORD")
    if username is not None:
        return username, password

    username = _env_or_dotenv(f"BORG_CAMERA_{index}_USERNAME") or ""
    password = _env_or_dotenv(f"BORG_CAMERA_{index}_PASSWORD")
    return username, password


class CameraManager:
    """Manage multiple RTSP camera clients from configuration."""

    def __init__(self) -> None:
        self._clients: dict[str, CameraClient] = {}
        self._build_clients()

    def _build_clients(self) -> None:
        configs = settings.cameras if isinstance(settings.cameras, list) else []
        for index, cfg in enumerate(configs, start=1):
            if not isinstance(cfg, dict):
                continue
            if not cfg.get("enabled", False):
                continue
            name = cfg.get("name") or f"camera_{index}"
            username, password = _camera_credentials(name, index)
            client = CameraClient(
                name=name,
                host=cfg.get("host", "127.0.0.1"),
                username=username,
                password=password,
                rtsp_port=cfg.get("rtsp_port", 554),
                rtsp_path=cfg.get("rtsp_path", "/stream1"),
                reconnect_seconds=cfg.get("reconnect_seconds", settings.camera_reconnect_seconds),
                stream_quality=cfg.get("stream_quality", settings.camera_stream_quality),
            )
            self._clients[name] = client

    @property
    def client_names(self) -> list[str]:
        return list(self._clients.keys())

    def get_client(self, name: str) -> Optional[CameraClient]:
        return self._clients.get(name)

    def start_all(self) -> None:
        for client in self._clients.values():
            client.start()

    def stop_all(self) -> None:
        for client in self._clients.values():
            client.stop()

    def status(self, name: Optional[str] = None) -> dict[str, Any] | list[dict[str, Any]]:
        if name:
            client = self._clients.get(name)
            return client.status() if client else {"name": name, "error": "Camera not found"}
        return [c.status() for c in self._clients.values()]

    def snapshot(self, name: str) -> Optional[bytes]:
        client = self._clients.get(name)
        return client.get_frame_bytes() if client else None

    def stream(self, name: str):
        client = self._clients.get(name)
        if client is None:
            raise ValueError(f"Camera '{name}' not found")
        return client.mjpeg_stream()


camera_manager = CameraManager()

# Backwards-compatible singleton used by legacy code/tests.
camera = camera_manager.get_client(camera_manager.client_names[0]) if camera_manager.client_names else None
