from typing import List, Any
from pydantic import BaseModel, Field
from agentine.llm import Message, FileMessage
from agentine.utils import Utility
from agentine.agent import Agent
from print9 import print9


class Chatbot(BaseModel):
    client: Any
    messages: List[Message] = Field(default_factory=list)

    def cli_run(self, stream: bool = False, display_stats: bool = False):
        print("Chatbot started. Type 'exit' to quit.")
        print()
        while True:
            query = input("YOU: ")
            if query.lower() == "exit":
                print("Chatbot session ended.")
                break

            if query == "--upload file":
                file_message = FileMessage.from_terminal()
                print9(f"{len(file_message.files)} images uploaded.")
                text = input("YOU: ")
                file_message.text = text
                self.messages.append(file_message)
            else:
                user_message = Message(content=query)
                self.messages.append(user_message)

            if stream:
                print()
                print9("BOT: ", end="", color="green", flush=True)
                accumulated_content = ""
                stats = {}
                for chunk in self.client.stream(messages=self.messages):
                    if chunk.content is not None:
                        print9(chunk.content, color="green", end="", flush=True)
                        accumulated_content += chunk.content
                    stats = chunk.stats

                print("\n")
                full_response = Message(
                    role="assistant",
                    content=accumulated_content,
                    stats=stats,
                )
                self.messages.append(full_response)

                if display_stats:
                    print9(stats, color="red")
                    print()
            else:
                response = self.client.work(messages=self.messages)
                self.messages.append(response)
                print()
                print9(f"BOT: {response.content}", color="green")
                print()

                if display_stats:
                    print9(response.stats, color="red")
                    print()