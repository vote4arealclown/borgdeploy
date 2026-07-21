"""RimWorld-style colony visualisation state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Room:
    id: str
    name: str
    x: float  # 0-100 grid
    y: float
    w: float
    h: float
    color: str
    icon: str


@dataclass
class Agent:
    id: str
    name: str
    x: float
    y: float
    target_room: str = "command"
    state: str = "idle"
    progress: float = 0.0


class ColonySim:
    """Tracks where the Borg agent is and what it is doing."""

    ROOMS: list[Room] = [
        Room("command", "Command Core", 45, 10, 12, 12, "#3b82f6", "🧠"),
        Room("sensor", "Market Sensor", 10, 40, 14, 10, "#10b981", "📡"),
        Room("lab", "Forecast Lab", 40, 40, 16, 10, "#8b5cf6", "🔬"),
        Room("archive", "Memory Archive", 72, 40, 14, 10, "#f59e0b", "📚"),
        Room("tower", "Monitor Tower", 45, 70, 12, 10, "#ef4444", "📊"),
        Room("hive", "Learning Hive", 82, 70, 12, 10, "#ec4899", "🍯"),
    ]

    def __init__(self) -> None:
        self.agent = Agent(id="borg-1", name="Borg Prime", x=50.0, y=16.0)
        self.phase: str = "idle"
        self.task: str = "waiting"
        self.symbol: str = ""
        self.last_event_id: int = 0

    def room_by_id(self, room_id: str) -> Room:
        for r in self.ROOMS:
            if r.id == room_id:
                return r
        return self.ROOMS[0]

    def set_phase(self, phase: str, symbol: str = "", task: str = "") -> None:
        self.phase = phase
        self.symbol = symbol
        self.task = task
        room_map = {
            "observe": "sensor",
            "plan": "lab",
            "act": "lab",
            "reflect": "archive",
            "monitor": "tower",
            "idle": "command",
        }
        self.agent.target_room = room_map.get(phase, "command")
        self.agent.state = phase

    def tick(self, dt: float = 0.05) -> None:
        """Move agent toward target room."""
        target = self.room_by_id(self.agent.target_room)
        tx = target.x + target.w / 2
        ty = target.y + target.h / 2
        dx = tx - self.agent.x
        dy = ty - self.agent.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1.0:
            self.agent.progress = min(1.0, self.agent.progress + dt * 2)
            return
        speed = min(2.0, dist)
        self.agent.x += (dx / dist) * speed
        self.agent.y += (dy / dist) * speed
        self.agent.progress = max(0.0, self.agent.progress - dt)

    def state(self) -> dict[str, Any]:
        return {
            "rooms": [r.__dict__ for r in self.ROOMS],
            "agent": self.agent.__dict__,
            "phase": self.phase,
            "task": self.task,
            "symbol": self.symbol,
        }


colony = ColonySim()