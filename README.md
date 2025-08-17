# agentine

agentine is a Python framework to build AI agents and chatbots using large language models (LLMs). It supports agents, groups of agents, synchronous and asynchronous calls, JSON and text responses, and includes a basic CLI chatbot.

## Installation

```bash
pip install agentine
```

## Features

- Compatible with multiple LLM providers and models
- Supports synchronous and asynchronous calls, including streaming
- Outputs responses as text or structured JSON
- Switch LLM providers instantly with a single line of code
- Easy to update configuration (provider, model, temperature, API key, etc.)

## Quick Start

> Note: agentine uses OpenAI as the _default_ LLM provider. If you prefer another provider, see the Advanced Usage section below.

### 1. Basic LLM: chat and chat_async

```python
from agentine.llm import LLM, Message

llm = LLM(api_key="your-openai-api-key")

messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Say hello in one short sentence."),
]

# chat (sync)
reply = llm.chat(messages)
print("LLM chat:", reply.content)

# chat_async (async)
import asyncio
async def run():
    reply_async = await llm.chat_async(messages)
    print("LLM chat async:", reply_async.content)

asyncio.run(run())
```

### 2. Basic Agent: work, stream, JSON, simple CLI chatbot

```python
from agentine.agent import Agent
from agentine.llm import Message

agent = Agent(instruction="You are a helpful assistant.")
agent.llm.api_key = "sk-your-openai-api-key"

past_conversation = [
    Message(role="user", content="Hello!"),
    Message(role="assistant", content="Hi there! How can I help?"),
]

# work
response = agent.work(
    query="Who is the first person walking on the Moon?",
    messages=past_conversation,
)
print("Agent work (text):", response.content)

# stream
for chunk in agent.stream(query="Write one short sentence about the Moon."):
    print(chunk.content, end="")
print()

# JSON agent
json_agent = Agent(
    instruction=(
        "You are a helpful assistant. Always respond with a JSON object "
        "with exactly two keys: 'first_name' and 'last_name'."
    ),
    response_format="json_object"
)
json_agent.llm.api_key = "sk-your-openai-api-key"
json_response = json_agent.work("Who is the first person walking on the Moon?")
print("Agent work (json):", json_response.data)

```

### 3. CLI chatbot (agent as client)

```python
from agentine.agent import Agent
from agentine.chatbot import Chatbot

agent = Agent(
    instruction="You are a helpful assistant."
)
agent.llm.provider = "openai"

chatbot = Chatbot(client=agent)
chatbot.cli_run()
```

## Advanced Usage

### 1. Async support

```python
from agentine.agent import Agent
import asyncio

agent = Agent(instruction="You are a helpful assistant.")
agent.llm.provider = "openai"

async def main():
    # work_async
    result = await agent.work_async("Say hello in 5 words.")
    print(result.content)

    # stream_async
    async for chunk in agent.stream_async(query="One short sentence about Mars."):
        print(chunk.content, end="")
    print()

asyncio.run(main())
```

### 2. Update LLM config on the fly

```python
from agentine.agent import Agent

agent = Agent(instruction="You are a helpful assistant.")

# Switch provider – model/api_key/base_url will auto-adjust from known provider config
agent.llm.provider = "gemini"

# Change model and sampling params on the fly
agent.llm.model = "gemini-2.5-flash"
agent.llm.temperature = 0.2

# Optionally set API key directly (overrides env var)
# agent.llm.api_key = "sk-..."

# Use immediately
print(agent.work("Give a 10-word haiku about oceans.").content)
```

### 3. Custom LLM config (params not built-in)

agentine allows extra LLM parameters (passed through to the provider). Examples:

```python
from agentine.agent import Agent

agent = Agent(instruction="You are a helpful assistant.")
agent.llm.provider = "openai"

# These fields are not predefined in LLM but are supported by many providers
agent.llm.top_p = 0.2
agent.llm.frequency_penalty = 0.3

result = agent.work("Write a single concise sentence about the Sun.")
print(result.content)
```

### 4. Change the default provider via `agentine.config`

You can change the framework-wide default provider before creating any `LLM` or `Agent` instances:

```python
from agentine import config

# Must be set BEFORE creating LLM/Agent instances
config.DEFAULT_LLM_PROVIDER = "gemini"

from agentine.llm import LLM

# New instances now default to the provider above
llm = LLM()  # defaults to Gemini now

# Provide API key via env (e.g., GEMINI_API_KEY) or set programmatically
# llm.api_key = "your-gemini-api-key"
```

This affects only newly created instances; existing ones keep their current provider.

## Configuration

agentine reads API keys from environment variables by default:

- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `PERPLEXITY_API_KEY`

You can also set API keys and providers programmatically on the agent's LLM instance.

## Contributing

Contributions are welcome. Please open issues or pull requests on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

If you have questions or need help, feel free to open an issue or contact the maintainer.