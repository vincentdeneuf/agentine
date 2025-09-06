from typing import Dict, Any, Optional, List, Union, Iterator, AsyncIterator
from pydantic import BaseModel, Field, PrivateAttr
from openai import OpenAI, AsyncOpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor

from agentine.config import DEFAULT_LLM_PROVIDER, KNOWN_LLM_PROVIDERS
from agentine.message import Message, FileMessage


class LLM(BaseModel):
    api_key: Optional[str] = Field(default=None)
    provider: str = Field(default=DEFAULT_LLM_PROVIDER)
    model: Optional[str] = None
    base_url: str = ""

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
        if "provider" not in data:
            data["provider"] = DEFAULT_LLM_PROVIDER

        super().__init__(**data)
        self._apply_provider_config(self.provider)
        self._init_clients()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

        if name == "provider":
            self._apply_provider_config(value)

        if name in {"provider", "api_key", "base_url"}:
            self._update_clients()

    def _apply_provider_config(self, provider: str) -> None:
        provider_key = str(provider).lower()
        provider_config = KNOWN_LLM_PROVIDERS.get(provider_key)

        if not provider_config:
            raise ValueError(
                f"422 - UNKNOWN PROVIDER: '{provider.upper()}'\n\n"
                f"Valid providers are: {', '.join(KNOWN_LLM_PROVIDERS.keys())}\n"
                f"If you want to use a custom provider, please set 'api_key' and 'base_url' manually instead."
            )

        super().__setattr__("model", provider_config.get("default_model"))
        super().__setattr__("api_key", provider_config.get("api_key"))
        super().__setattr__("base_url", provider_config.get("base_url"))

    def _init_clients(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._client_async = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _update_clients(self) -> None:
        self._client.api_key = self.api_key
        self._client.base_url = self.base_url
        self._client_async.api_key = self.api_key
        self._client_async.base_url = self.base_url

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