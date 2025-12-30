from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass
class Event:
    date: pd.Timestamp
    action: str
    manager: str
    reason: str
    details: dict[str, Any] | None = None


class EventLog:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def append(self, e: Event) -> None:
        self.events.append(e)

    def to_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(
                columns=["date", "action", "manager", "reason", "details"]
            ).set_index("date")
        df = pd.DataFrame([asdict(e) for e in self.events])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()
