from typing import Dict, Any, Optional, List, Union, Literal, Iterator, AsyncIterator
from pydantic import BaseModel, Field, PrivateAttr
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
import base64
import mimetypes
import asyncio
from concurrent.futures import ThreadPoolExecutor

from agentine.metadata import Metadata, Stats
from agentine.utils import Utility
from agentine.config import DEFAULT_LLM_PROVIDER, KNOWN_LLM_PROVIDERS
from print9 import print9


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
    def from_openai_completion(cls, completion: "ChatCompletion") -> "Message":
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
    def from_openai_completion_chunk(cls, chunk: "ChatCompletionChunk") -> "Message":
        chunk_dict = (
            chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
        )
        usage = chunk.usage or None
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

class LLM(BaseModel):
    api_key: Optional[str] = Field(default=None)
    provider: str = Field(default=DEFAULT_LLM_PROVIDER)
    model: Optional[str] = None
    base_url: Optional[str] = None

    timeout: int = 60000
    max_retries: int = 2
    max_concurrency: int = 100

    response_format: str = Field(default="text")
    temperature: float = 1.0
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None

    _client: OpenAI = PrivateAttr()
    _client_async: AsyncOpenAI = PrivateAttr()

    class Config:
        extra = "allow"

    def __init__(self, **data: Any):
        provider = data.get("provider", DEFAULT_LLM_PROVIDER)
        provider_config = KNOWN_LLM_PROVIDERS.get(provider, {})

        provider_models = provider_config.get("models", {})
        default_model = provider_models.get("default")
        default_api_key = provider_config.get("api_key")
        default_base_url = provider_config.get("base_url")

        data.setdefault("provider", provider)
        data.setdefault("model", default_model)
        data.setdefault("api_key", default_api_key)
        data.setdefault("base_url", default_base_url)

        super().__init__(**data)

        self._init_clients()

    def __setattr__(self, name: str, value: Any) -> None:
        current_value = getattr(self, name, None)
        if current_value == value:
            return

        super().__setattr__(name, value)
        special_attributes = {"provider", "api_key", "base_url"}
        if name in special_attributes:
            if name == "provider" and value in KNOWN_LLM_PROVIDERS:
                provider_config = KNOWN_LLM_PROVIDERS[value]
                provider_models = provider_config.get("models", {})
                default_model = provider_models.get("default")

                self.__dict__["model"] = default_model
                self.__dict__["api_key"] = provider_config.get("api_key")
                self.__dict__["base_url"] = provider_config.get("base_url")
            self._init_clients()

    def _init_clients(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._client_async = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def client(self) -> OpenAI:
        return self._client

    @property
    def client_async(self) -> AsyncOpenAI:
        return self._client_async

    def completion_config(self) -> Dict[str, Any]:
        excluded = {
            "api_key",
            "provider",
            "base_url",
            "timeout",
            "max_retries",
            "max_concurrency",
        }
        data = self.model_dump(exclude=excluded, exclude_none=True)
        if "response_format" in data:
            data["response_format"] = {"type": data["response_format"]}
        return data

    def _prepare_kwargs(self, messages: List["Message"]) -> Dict[str, Any]:
        assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages)
        kwargs = self.completion_config()
        kwargs["model"] = self.model
        kwargs["messages"] = [m.core() for m in messages]
        return kwargs

    def chat(self, messages: List["Message"]) -> "Message":
        kwargs = self._prepare_kwargs(messages)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return Message.from_openai_completion(response)
            except Exception:
                if attempt == self.max_retries:
                    raise

    async def chat_async(self, messages: List["Message"]) -> "Message":
        kwargs = self._prepare_kwargs(messages)
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client_async.chat.completions.create(**kwargs)
                return Message.from_openai_completion(response)
            except Exception:
                if attempt == self.max_retries:
                    raise

    def stream(
        self, messages: List["Message"], include_usage: bool = True
    ) -> Iterator["Message"]:
        kwargs = self._prepare_kwargs(messages)
        kwargs["stream"] = True
        if include_usage:
            kwargs["stream_options"] = {"include_usage": True}
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            yield Message.from_openai_completion_chunk(chunk)

    async def stream_async(
        self, messages: List["Message"], include_usage: bool = True
    ) -> AsyncIterator["Message"]:
        kwargs = self._prepare_kwargs(messages)
        kwargs["stream"] = True
        if include_usage:
            kwargs["stream_options"] = {"include_usage": True}
        stream = await self.client_async.chat.completions.create(**kwargs)
        async for chunk in stream:
            yield Message.from_openai_completion_chunk(chunk)

    def batch(self, batch_messages: List[List["Message"]]) -> List[Union["Message", Exception]]:
        def process(messages: List["Message"]) -> Union["Message", Exception]:
            try:
                return self.chat(messages)
            except Exception as e:
                return e
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            return list(executor.map(process, batch_messages))

    async def batch_async(self, batch_messages: List[List["Message"]]) -> List[Union["Message", Exception]]:
        async def process(messages: List["Message"]) -> Union["Message", Exception]:
            try:
                return await self.chat_async(messages)
            except Exception as e:
                return e
        return await asyncio.gather(*(process(m) for m in batch_messages), return_exceptions=True)

