from agentine.agent import Agent
from agentine.chatbot import Chatbot
from agentine.llm import LLM
from agentine.llm import Message

if __name__ == "__main__":

    general_agent = Agent(
        instruction="You are a helpful assistant. Answer the user's questions clearly and concisely.",
    )
    general_agent.llm.provider = "openai"
    general_agent.llm.provider = "gemini"
    # result = general_agent.work(query="hello, how are you today?", messages=[])
    # print(result)


    chatbot = Chatbot(client=general_agent)
    chatbot.cli_run(stream=True)