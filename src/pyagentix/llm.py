from typing import Dict, Any, Optional, List, Union, Literal, Iterator, AsyncIterator
from pydantic import BaseModel, Field, PrivateAttr
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

import base64
import mimetypes
import asyncio
from concurrent.futures import ThreadPoolExecutor

from pyagentix.metadata import Metadata
from pyagentix.utils import Utility
from pyagentix.config import DEFAULT_LLM_PROVIDER


class Message(BaseModel):
    role: Optional[Literal["system", "developer", "user", "assistant", "tool"]] = "user"
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    data: Optional[Any] = None
    choice_stats: Optional[Dict[str, Any]] = None
    completion_stats: Optional[Dict[str, Any]] = None
    metadata: Metadata = Field(default_factory=Metadata)

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
    def from_openai_completion(cls, completion: ChatCompletion) -> "Message":
        completion_dict = completion.model_dump() if hasattr(completion, "model_dump") else dict(completion)
        choices = completion_dict.pop("choices", [])
        choice_dict = choices[0] if choices else {}
        message = choice_dict.get("message", {})
        content = message.pop("content", "") if isinstance(message, dict) else ""
        message.pop("role")
        role = "assistant"

        return cls(
            content=content,
            role=role,
            choice_stats=choice_dict,
            completion_stats=completion_dict,
        )

    @classmethod
    def from_openai_completion_chunk(cls, chunk: "ChatCompletionChunk") -> "Message":
        chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
        choices = chunk_dict.pop("choices", [])
        choice_dict = choices[0] if choices else {}
        delta = choice_dict.get("delta", {})
        content = delta.pop("content", "") if isinstance(delta, dict) else ""
        delta.pop("role")
        role = "assistant"

        message = cls(
            role=role,
            content=content,
            choice_stats=choice_dict,
            completion_stats=chunk_dict,
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
    class Config(BaseModel):
        response_format: Literal["json_schema", "json_object", "text"] = Field(default="text")
        temperature: float = Field(default=1)
        max_retries: int = Field(default=2)
        timeout: int = Field(default=60000)
        max_completion_tokens: int = Field(default=None)
        max_concurrency: int = Field(default=100)
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(default=None)

        def core(self) -> Dict[str, Any]:
            included_fields = {
                "temperature",
                "max_completion_tokens",
                "response_format",
                "reasoning_effort",
            }
            data = self.model_dump(include=included_fields, exclude_none=True)
            if "response_format" in data:
                data["response_format"] = {"type": data["response_format"]}
            return data

    _api_key: Optional[str] = PrivateAttr(default=None)
    _provider: Literal["openai", "groq", "gemini"] = PrivateAttr(default=DEFAULT_LLM_PROVIDER)

    _client: OpenAI = PrivateAttr()
    _client_async: AsyncOpenAI = PrivateAttr()

    model: Optional[str] = Field(default=None)
    config: "LLM.Config" = Field(default_factory=Config)

    def __init__(self, **data: Any):
        super().__init__(**data)
        if 'api_key' in data:
            self._api_key = data['api_key']
        if 'provider' in data:
            self._provider = data['provider']
        self._init_clients()

    def _init_clients(self) -> None:
        from pyagentix.config import LLM_PROVIDERS  # import here to avoid circular import

        provider_config = LLM_PROVIDERS.get(self._provider.lower())

        base_url = provider_config.get("base_url")
        actual_api_key = self._api_key or provider_config.get("api_key")
        provider_models = provider_config.get("models")
        default_provider_model = provider_models.get("default")

        if self.model is None:
            self.model = default_provider_model

        elif self.model not in provider_models.values():
            self.model = default_provider_model

        self._client = OpenAI(api_key=actual_api_key, base_url=base_url)
        self._client_async = AsyncOpenAI(api_key=actual_api_key, base_url=base_url)

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        if self._api_key != value:
            self._api_key = value
            self._init_clients()

    @property
    def provider(self):
        return self._provider

    @provider.setter
    def provider(self, value):
        if not isinstance(value, str) or value not in ["openai", "groq", "gemini"]:
            raise ValueError(f"Provider must be one of 'openai', 'groq', 'gemini'. Got: {value}")
        self._provider = value
        self._init_clients()

    @property
    def client(self) -> OpenAI:
        return self._client

    @property
    def client_async(self) -> AsyncOpenAI:
        return self._client_async

    def chat(self, messages: List["Message"]) -> "Message":
        assert isinstance(messages, list) and all(isinstance(message, Message) for message in messages), \
            f"messages must be a list of Message objects. Value: {messages}"

        openai_messages = [message.core() for message in messages]

        kwargs: Dict[str, Any] = self.config.core()
        kwargs["model"] = self.model
        kwargs["messages"] = openai_messages

        for attempt in range(self.config.max_retries + 1):
            try:
                response: ChatCompletion = self.client.chat.completions.create(**kwargs)
                return Message.from_openai_completion(response)
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise e

    async def chat_async(self, messages: List["Message"]) -> "Message":
        assert isinstance(messages, list) and all(isinstance(message, Message) for message in messages), \
            f"messages must be a list of Message objects. Value: {messages}"

        openai_messages = [message.core() for message in messages]

        kwargs: Dict[str, Any] = self.config.core()
        kwargs["model"] = self.model
        kwargs["messages"] = openai_messages

        for attempt in range(self.config.max_retries + 1):
            try:
                response: ChatCompletion = await self.client_async.chat.completions.create(**kwargs)
                return Message.from_openai_completion(response)
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise e

    def batch(self, batch_messages: List[List["Message"]]) -> List[Union["Message", Exception]]:
        assert isinstance(batch_messages, list) and all(
            isinstance(messages, list) and all(isinstance(message, Message) for message in messages)
            for messages in batch_messages
        ), f"batch_messages must be a list of lists of Message objects. Value: {batch_messages}"

        def process_message(messages: List["Message"]) -> Union["Message", Exception]:
            try:
                return self.chat(messages)
            except Exception as e:
                return e

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            results: List[Union["Message", Exception]] = list(executor.map(process_message, batch_messages))
        return results

    async def batch_async(self, batch_messages: List[List["Message"]]) -> List[Union["Message", Exception]]:
        assert isinstance(batch_messages, list) and all(
            isinstance(messages, list) and all(isinstance(message, Message) for message in messages)
            for messages in batch_messages
        ), f"batch_messages must be a list of lists of Message objects. Value: {batch_messages}"

        async def process_message_async(messages: List["Message"]) -> Union["Message", Exception]:
            try:
                return await self.chat_async(messages)
            except Exception as e:
                return e

        tasks = [process_message_async(messages) for messages in batch_messages]
        results: List[Union["Message", Exception]] = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def stream(self, messages: List["Message"]) -> Iterator["Message"]:
        assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages), \
            f"messages must be a list of Message objects. Value: {messages}"

        openai_messages = [m.core() for m in messages]

        kwargs = self.config.core()
        kwargs["model"] = self.model
        kwargs["messages"] = openai_messages
        kwargs["stream"] = True

        stream = self.client.chat.completions.create(**kwargs)

        for chunk in stream:
            yield Message.from_openai_completion_chunk(chunk)

    async def stream_async(self, messages: List["Message"]) -> AsyncIterator["Message"]:
        assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages), \
            f"messages must be a list of Message objects. Value: {messages}"

        openai_messages = [m.core() for m in messages]

        kwargs = self.config.core()
        kwargs["model"] = self.model
        kwargs["messages"] = openai_messages
        kwargs["stream"] = True

        stream = await self.client_async.chat.completions.create(**kwargs)

        async for chunk in stream:
            yield Message.from_openai_completion_chunk(chunk)
        