# PyAgentix

PyAgentix is a Python framework to build AI agents and chatbots using large language models (LLMs). It supports agents, groups of agents, synchronous and asynchronous calls, JSON and text responses, and includes a basic CLI chatbot.

---

## Installation

```bash
pip install pyagentix
```

*Note:* This package requires Python 3.8+.

---

## Features

- Agent and AgentGroup abstractions
- Synchronous and asynchronous support
- JSON and text response formats
- CLI chatbot with file upload support

---

## Quick Start

### 1. Create and use a text response agent

```python
from pyagentix.agent import AgentUnit

agent = AgentUnit(
    instruction="You are a helpful assistant."
)
agent.llm.provider = "openai"

response = agent.work("Who is the first person walking on the Moon?")
print("Text response:", response.content)
```

---

### 2. Create and use a JSON response agent

```python
from pyagentix.agent import AgentUnit

json_agent = AgentUnit(
    instruction=(
        "You are a helpful assistant. Always respond with a JSON object "
        "with exactly two keys: 'first_name' and 'last_name'."
    ),
    response_format="json_object"
)
json_agent.llm.provider = "openai"

json_response = json_agent.work("Who is the first person walking on the Moon?")
print("JSON response:", json_response.data)
```

---

### 3. Use the agent asynchronously

```python
import asyncio

query = "Who is the first person walking on the Moon?"
text_response = asyncio.run(agent.work_async(query))
print("Async text response:", text_response.content)
```

---

### 4. Run the CLI chatbot

```python
from pyagentix.agent import AgentUnit
from pyagentix.chatbot import Chatbot

agent = AgentUnit(
    instruction="You are a helpful assistant."
)
agent.llm.provider = "openai"

chatbot = Chatbot(client=agent)
chatbot.cli_run()
```

---

## Configuration

PyAgentix reads API keys from environment variables by default:

- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `PERPLEXITY_API_KEY`

You can also set API keys and providers programmatically on the agent's LLM instance.

---

## Contributing

Contributions are welcome. Please open issues or pull requests on GitHub.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

If you have questions or need help, feel free to open an issue or contact the maintainer.