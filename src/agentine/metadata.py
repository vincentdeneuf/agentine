from datetime import datetime
from pydantic import BaseModel, Field


class ChangeLog(BaseModel):
    time: datetime = Field(default_factory=datetime.utcnow)
    fields: list[str] = Field(default_factory=list, min_items=1)


class Metadata(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    change_logs: list[ChangeLog] = Field(default_factory=list)

    class Config:
        extra = "allow"

    def __setattr__(self, name: str, value: object) -> None:
        if name == "created_at":
            if hasattr(self, name):
                raise ValueError(f"'{name}' field is immutable and cannot be changed.")
        super().__setattr__(name, value)

    def log_change(self, fields: list[str]) -> None:
        self.change_logs.append(ChangeLog(fields=fields))


class Stats(BaseModel):
    class Config:
        extra = "allow"