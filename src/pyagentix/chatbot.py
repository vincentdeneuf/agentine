from typing import List, Any
from pydantic import BaseModel, Field
from pyagentix.llm import Message, FileMessage
from pyagentix.utils import Utility
from pyagentix.agent import AgentUnit

class Chatbot(BaseModel):
    client: Any
    messages: List[Message] = Field(default_factory=list)

    def cli_run(self, stream: bool = False):
        print("Chatbot started. Type 'exit' to quit.")
        print()
        while True:
            query = input("YOU: ")
            if query.lower() == "exit":
                print("Chatbot session ended.")
                break

            if query == "--upload file":
                file_message = FileMessage.from_terminal()
                Utility.print2(f"{len(file_message.files)} images uploaded.")
                text = input("YOU: ")
                file_message.text = text
                self.messages.append(file_message)
            else:
                user_message = Message(content=query)
                self.messages.append(user_message)

            if stream:
                print()
                print("BOT: ", end="", flush=True)
                accumulated_content = ""
                for chunk in self.client.stream(messages=self.messages):
                    if chunk.content:
                        Utility.print2(chunk.content, color = "green", end="", flush=True)
                        accumulated_content += chunk.content
                print("\n")
                full_response = Message(role="assistant", content=accumulated_content)
                self.messages.append(full_response)
            else:
                response = self.client.work(messages=self.messages)
                self.messages.append(response)
                print()
                Utility.print2(f"BOT: {response.content}", color="green")
                print()
