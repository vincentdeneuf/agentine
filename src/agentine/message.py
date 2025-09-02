from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import BaseModel, Field
import base64
import mimetypes

from agentine.metadata import Metadata, Stats
from agentine.utils import Utility


class Message(BaseModel):
    role: Optional[Literal["system", "developer", "user", "assistant", "tool"]] = "user"
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    data: Optional[Any] = None

    stats: Stats = Field(default_factory=Stats)
    metadata: Metadata = Field(default_factory=Metadata)

    class Config:
        extra = "allow"

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "metadata":
            old_value = getattr(self, name, None)
            super().__setattr__(name, value)
            if old_value != value:
                self.metadata.log_change(fields=[name])
        else:
            super().__setattr__(name, value)

    def core(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_openai_completion(cls, completion: Any) -> "Message":
        completion_dict = (
            completion.model_dump()
            if hasattr(completion, "model_dump")
            else dict(completion)
        )
        choices = completion_dict.pop("choices", [])
        choice_dict = choices[0] if choices else {}
        message = choice_dict.get("message", {})
        content = message.pop("content", "") if isinstance(message, dict) else ""
        message.pop("role", None)
        role = "assistant"

        return cls(
            content=content,
            role=role,
            stats=Stats(choice=choice_dict, completion=completion_dict),
        )

    @classmethod
    def from_openai_completion_chunk(cls, chunk: Any) -> "Message":
        chunk_dict = (
            chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
        )
        usage = getattr(chunk, "usage", None)
        choices = chunk_dict.pop("choices", [])
        choice_dict = choices[0] if choices else {}
        delta = choice_dict.get("delta", {})
        content = delta.pop("content", "") if isinstance(delta, dict) else ""
        delta.pop("role", None)
        role = "assistant"

        message = cls(
            role=role,
            content=content,
            stats=Stats(choice=choice_dict, completion=chunk_dict, usage=usage),
        )
        message.metadata.is_chunk = True

        return message


class FileMessage(Message):
    text: str = ""
    files: List[Dict[str, Any]] = Field(default_factory=list)
    content: List[Dict] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self._update_content()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in {"text", "files"}:
            self._update_content()

    def _update_content(self):
        content_blocks = []
        if self.text:
            content_blocks.append({"type": "text", "text": self.text})

        for file_info in self.files:
            filename = file_info.get("filename")
            data_url = file_info.get("data_url")
            mime_type = file_info.get("mime_type", "application/octet-stream")

            if mime_type.startswith("image/"):
                content_blocks.append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )
            else:
                content_blocks.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": data_url,
                        },
                    }
                )

        super().__setattr__("content", content_blocks)

    def core(self) -> dict:
        return {"role": self.role, "content": self.content}

    @staticmethod
    def from_terminal(text: str = "") -> "FileMessage":
        file_path = Utility.get_file_path_via_terminal()
        if not file_path:
            raise ValueError("No file selected")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        filename = file_path.split("/")[-1]
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None:
            mime_type = "application/octet-stream"

        b64_str = base64.b64encode(file_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_str}"

        files_list = [
            {
                "filename": filename,
                "data_url": data_url,
                "mime_type": mime_type,
            }
        ]

        return FileMessage(text=text, files=files_list)