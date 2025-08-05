from datetime import datetime
from typing import List, Any
from pydantic import BaseModel, Field

class ChangeLog(BaseModel):
    time: datetime = Field(default_factory=datetime.utcnow)
    fields: List[str] = Field(default_factory=list, min_items=1)

class Metadata(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    change_logs: List[ChangeLog] = Field(default_factory=list)

    class Config:
        extra = "allow"

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "created_at":
            if hasattr(self, name):
                raise ValueError(f"'{name}' field is immutable and cannot be changed.")
        super().__setattr__(name, value)

    def log_change(self, fields: List[str]) -> None:
        self.change_logs.append(ChangeLog(fields=fields))