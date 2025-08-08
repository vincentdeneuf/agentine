import json
from typing import AsyncIterator, Iterator, List, Optional, Dict, Any, Tuple, Literal
from pydantic import BaseModel, Field
import asyncio
from pyagentix.llm import LLM, Message
from pyagentix.utils import Utility

class AgentUnit(BaseModel):
    instruction: str
    group_instruction: Optional[str] = None
    llm: LLM = Field(default_factory=LLM)
    output_label: str = "output"
    response_format: Literal["text", "json_schema", "json_object"] = Field(default="text")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.llm.config.response_format = self.response_format

    def _prepare_messages(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        data: Optional[Dict[str, Any]] = None,
        group_mode: bool = False,
    ) -> List[Message]:
        if data is not None:
            assert "output_label" not in data, "Agent data cannot contain reserved key 'output_label'."
            data["output_label"] = self.output_label

        selected_instruction = self.instruction
        if group_mode and self.group_instruction:
            selected_instruction = self.group_instruction

        formatted_instruction = (
            Utility.format(string=selected_instruction, data=data) if data else selected_instruction
        )
        formatted_query = Utility.format(string=query, data=data) if data else query

        chat_messages: List[Message] = []

        if formatted_instruction:
            chat_messages.append(Message(role="system", content=formatted_instruction))
        if messages:
            chat_messages.extend(messages)
        if formatted_query:
            chat_messages.append(Message(role="user", content=formatted_query))

        return chat_messages

    def work(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        data: Optional[Dict[str, Any]] = None,
        keep_output_label: bool = False,
        group_mode: bool = False,
    ) -> Message:
        assert query is None or isinstance(query, str), "query must be a string or None"
        assert messages is None or isinstance(messages, list), "messages must be a list or None"
        assert data is None or isinstance(data, dict), "data must be a dict or None"

        chat_messages = self._prepare_messages(query=query, messages=messages, data=data, group_mode=group_mode)

        result = self.llm.chat(messages=chat_messages)

        if self.response_format != "text":
            parsed_data = json.loads(result.content)
            wrapped = Utility.wrap(data=parsed_data, key=self.output_label)
            if keep_output_label:
                result.data = wrapped
            else:
                result.data = wrapped[self.output_label]

        return result

    async def work_async(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        data: Optional[Dict[str, Any]] = None,
        keep_output_label: bool = False,
        group_mode: bool = False,
    ) -> Message:
        assert query is None or isinstance(query, str), "query must be a string or None"
        assert messages is None or isinstance(messages, list), "messages must be a list or None"
        assert data is None or isinstance(data, dict), "data must be a dict or None"

        chat_messages = self._prepare_messages(query=query, messages=messages, data=data, group_mode=group_mode)

        result = await self.llm.chat_async(messages=chat_messages)

        if self.response_format != "text":
            parsed_data = json.loads(result.content)
            wrapped = Utility.wrap(data=parsed_data, key=self.output_label)
            if keep_output_label:
                result.data = wrapped
            else:
                result.data = wrapped[self.output_label]

        return result

    def stream(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        data: Optional[Dict[str, Any]] = None,
        group_mode: bool = False,
    ) -> Iterator[Message]:
        chat_messages = self._prepare_messages(query=query, messages=messages, data=data, group_mode=group_mode)
        return self.llm.stream(chat_messages)

    async def stream_async(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        data: Optional[Dict[str, Any]] = None,
        group_mode: bool = False,
    ) -> AsyncIterator[Message]:
        chat_messages = self._prepare_messages(query=query, messages=messages, data=data, group_mode=group_mode)
        async for message in self.llm.stream_async(chat_messages):
            yield message

class AgentGroup(BaseModel):
    agents: List[AgentUnit] = Field(default_factory=list)

    def work(
        self,
        query: str,
        messages: Optional[List[Message]] = None,
        data: Optional[Dict[str, Any]] = None,
        keep_output_label: bool = False,
        group_mode: bool = False,
    ) -> List[Message]:
        results = []
        for agent in self.agents:
            result = agent.work(
                query=query,
                messages=messages,
                data=data,
                keep_output_label=keep_output_label,
                group_mode=group_mode
            )
            results.append(result)
        return results

    async def work_async(
        self,
        query: str,
        messages: Optional[List[Message]] = None,
        data: Optional[Dict[str, Any]] = None,
        keep_output_label: bool = False,
        group_mode: bool = False,
    ) -> List[Message]:
        tasks = [
            agent.work_async(
                query=query,
                messages=messages,
                data=data,
                keep_output_label=keep_output_label,
                group_mode=group_mode
            )
            for agent in self.agents
        ]
        results = await asyncio.gather(*tasks)
        return results


class AgentIndex(BaseModel):
    agents: Dict[str, AgentUnit] = Field(default_factory=dict)
    default: Optional[AgentUnit] = None

    def __getitem__(self, key: str) -> AgentUnit:
        return self.agents[key]

    def add(self, name: str, agent: AgentUnit, is_default: bool = False) -> None:
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

    def find(self, names: List[str]) -> List[AgentUnit]:
        return [agent for name in names if (agent := self.agents.get(name)) is not None]


class AgentLegion(BaseModel):
    speaker: AgentUnit
    selector: AgentUnit
    agent_index: AgentIndex

    def _prepare_messages(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Message]] = None,
    ) -> Tuple[List[Message], Optional[AgentUnit]]:
        if messages is None:
            messages = []

        selector_result = self.selector.work(query=query, messages=messages)

        assert selector_result.data is not None, "Selector result missing data."
        assert "selections" in selector_result.data, "Selector data missing 'selections' key."

        selections = selector_result.data["selections"]
        Utility.print2(f"{selections}")

        selected_agents = self.agent_index.find(selections)

        if len(selected_agents) == 1:
            return messages, selected_agents[0]

        agent_group = AgentGroup(agents=selected_agents)

        results = asyncio.run(
            agent_group.work_async(query=query, messages=messages, group_mode=True)
        )

        for agent_name, result in zip(selections, results):
            result.content = (
                f"**{agent_name} agent** response (NOT VISIBLE TO USER):\n\n{result.content}"
            )
            Utility.print2(result.content)
            print()
            messages.append(result)

        return messages, None

    def work(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Message]] = None,
    ) -> Message:
        messages, single_agent = self._prepare_messages(query=query, messages=messages)

        if single_agent is not None:
            return single_agent.work(query=query, messages=messages)

        return self.speaker.work(messages=messages)

    def stream(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Message]] = None,
    ) -> Iterator[Message]:
        messages, single_agent = self._prepare_messages(query=query, messages=messages)

        if single_agent is not None:
            yield from single_agent.stream(query=query, messages=messages)
            return

        yield from self.speaker.stream(messages=messages)