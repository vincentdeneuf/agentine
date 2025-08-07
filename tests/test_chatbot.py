import pytest
from pyagentix.agent import AgentUnit
from pyagentix.chatbot import Chatbot
from pyagentix.llm import LLM
from pyagentix.llm import Message

if __name__ == "__main__":

    general_agent = AgentUnit(
        instruction="You are a helpful assistant. Answer the user's questions clearly and concisely.",
    )
    general_agent.llm.provider = "gemini"
    chatbot = Chatbot(client=general_agent)
    chatbot.cli_run(stream=True)