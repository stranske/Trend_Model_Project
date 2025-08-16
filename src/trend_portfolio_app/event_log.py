from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd


@dataclass
class Event:
    date: pd.Timestamp
    action: str
    manager: str
    reason: str
    details: Optional[Dict[str, Any]] = None


class EventLog:
    def __init__(self):
        self.events: List[Event] = []

    def append(self, e: Event):
        self.events.append(e)

    def to_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(
                columns=["date", "action", "manager", "reason", "details"]
            ).set_index("date")
        df = pd.DataFrame([asdict(e) for e in self.events])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()
