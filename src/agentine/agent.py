import json
from collections.abc import AsyncIterator, Iterator
from typing import Literal
from pydantic import BaseModel, Field
import asyncio
from agentine.llm import LLM, Message
from agentine.utils import Utility


class Agent(BaseModel):
    instruction: str
    name: str | None = None
    llm: LLM = Field(default_factory=LLM)
    response_format: Literal["text", "json_schema", "json_object"] = Field(default="text")

    class Config:
        extra = "allow"

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self.llm.response_format = self.response_format

    def __setattr__(self, key: str, value: object) -> None:
        super().__setattr__(key, value)
        if key == "response_format" and hasattr(self, "llm") and self.llm is not None:
            self.llm.response_format = value

    def _prepare_messages(
        self,
        query: str | None = None,
        messages: list[Message] | None = None,
        data: dict[str, object] | None = None,
    ) -> list[Message]:
        formatted_instruction = Utility.format(string=self.instruction, data=data) if data else self.instruction
        formatted_query = Utility.format(string=query, data=data) if data else query

        chat_messages: list[Message] = []

        if formatted_instruction:
            chat_messages.append(Message(role="system", content=formatted_instruction))
        if messages:
            chat_messages.extend(messages)
        if formatted_query:
            chat_messages.append(Message(role="user", content=formatted_query))

        return chat_messages

    def work(
        self,
        query: str | None = None,
        messages: list[Message] | None = None,
        data: dict[str, object] | None = None,
    ) -> Message:
        assert query is None or isinstance(query, str), "query must be a string or None"
        assert messages is None or isinstance(messages, list), "messages must be a list or None"
        assert data is None or isinstance(data, dict), "data must be a dict or None"

        chat_messages = self._prepare_messages(query=query, messages=messages, data=data)
        result = self.llm.chat(messages=chat_messages)

        if self.response_format == "json_object":
            result.data = json.loads(result.content)

        return result

    async def work_async(
        self,
        query: str | None = None,
        messages: list[Message] | None = None,
        data: dict[str, object] | None = None,
    ) -> Message:
        assert query is None or isinstance(query, str), "query must be a string or None"
        assert messages is None or isinstance(messages, list), "messages must be a list or None"
        assert data is None or isinstance(data, dict), "data must be a dict or None"

        chat_messages = self._prepare_messages(query=query, messages=messages, data=data)
        result = await self.llm.chat_async(messages=chat_messages)

        if self.response_format == "json_object":
            result.data = json.loads(result.content)

        return result

    def stream(
        self,
        query: str | None = None,
        messages: list[Message] | None = None,
        data: dict[str, object] | None = None,
    ) -> Iterator[Message]:
        assert query is None or isinstance(query, str), "query must be a string or None"
        assert messages is None or isinstance(messages, list), "messages must be a list or None"
        assert data is None or isinstance(data, dict), "data must be a dict or None"

        chat_messages = self._prepare_messages(query=query, messages=messages, data=data)
        return self.llm.stream(chat_messages)

    async def stream_async(
        self,
        query: str | None = None,
        messages: list[Message] | None = None,
        data: dict[str, object] | None = None,
    ) -> AsyncIterator[Message]:
        assert query is None or isinstance(query, str), "query must be a string or None"
        assert messages is None or isinstance(messages, list), "messages must be a list or None"
        assert data is None or isinstance(data, dict), "data must be a dict or None"

        chat_messages = self._prepare_messages(query=query, messages=messages, data=data)
        async for message in self.llm.stream_async(chat_messages):
            yield message


class AgentGroup(BaseModel):
    agents: list[Agent] = Field(default_factory=list)

    def work(
        self,
        query: str,
        messages: list[Message] | None = None,
        data: dict[str, object] | None = None,
    ) -> list[Message]:
        results = []
        for agent in self.agents:
            result = agent.work(query=query, messages=messages, data=data)
            results.append(result)
        return results

    async def work_async(
        self,
        query: str,
        messages: list[Message] | None = None,
        data: dict[str, object] | None = None,
    ) -> list[Message]:
        tasks = [
            agent.work_async(query=query, messages=messages, data=data)
            for agent in self.agents
        ]
        results = await asyncio.gather(*tasks)
        return results


class AgentIndex(BaseModel):
    agents: dict[str, Agent] = Field(default_factory=dict)
    default: Agent | None = None

    def __getitem__(self, key: str) -> Agent:
        return self.agents[key]

    def add(self, name: str, agent: Agent, is_default: bool = False) -> None:
        self.agents[name] = agent
        if is_default:
            self.default = agent

    def remove(self, name: str) -> None:
        if name not in self.agents:
            raise KeyError(f"Agent '{name}' not found in the index.")

        if self.default == self.agents[name]:
            self.default = None

        del self.agents[name]

    def set_default(self, name: str) -> None:
        if name not in self.agents:
            raise KeyError(f"Agent '{name}' not found in the index.")
        self.default = self.agents[name]

    def find(self, names: list[str]) -> list[Agent]:
        return [agent for name in names if (agent := self.agents.get(name)) is not None]


class AgentLegion(BaseModel):
    speaker: Agent
    selector: Agent
    agent_index: AgentIndex

    def _prepare_messages(
        self,
        query: str | None = None,
        messages: list[Message] | None = None,
    ) -> tuple[list[Message], Agent | None]:
        if messages is None:
            messages = []

        selector_result = self.selector.work(query=query, messages=messages)

        assert selector_result.data is not None, "Selector result missing data."
        assert "selections" in selector_result.data, "Selector data missing 'selections' key."

        selections = selector_result.data["selections"]
        selected_agents = self.agent_index.find(selections)

        if len(selected_agents) == 1:
            return messages, selected_agents[0]

        agent_group = AgentGroup(agents=selected_agents)

        results = asyncio.run(
            agent_group.work_async(query=query, messages=messages)
        )

        for agent_name, result in zip(selections, results):
            result.content = (
                f"**{agent_name} agent** response (NOT VISIBLE TO USER):\n\n{result.content}"
            )
            messages.append(result)

        return messages, None

    def work(
        self,
        query: str | None = None,
        messages: list[Message] | None = None,
    ) -> Message:
        messages, single_agent = self._prepare_messages(query=query, messages=messages)

        if single_agent is not None:
            return single_agent.work(query=query, messages=messages)

        return self.speaker.work(messages=messages)

    def stream(
        self,
        query: str | None = None,
        messages: list[Message] | None = None,
    ) -> Iterator[Message]:
        messages, single_agent = self._prepare_messages(query=query, messages=messages)

        if single_agent is not None:
            yield from single_agent.stream(query=query, messages=messages)
            return

        yield from self.speaker.stream(messages=messages)