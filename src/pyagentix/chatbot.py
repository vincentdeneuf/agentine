from typing import List, Any
from pydantic import BaseModel, Field
from .llm import Message, FileMessage
from .utils import Utility

class Chatbot(BaseModel):
    client: Any
    messages: List[Message] = Field(default_factory=list)

    def cli_run(self):
        print("Chatbot started. Type 'exit' to quit.")
        print()
        while True:
            query = input("YOU: ")
            if query.lower() == "exit":
                print("Chatbot session ended.")
                break

            if query == "--upload file":
                file_message = FileMessage.from_upload()
                Utility.print2(f"{len(file_message.files)} images uploaded.")
                text = input("YOU: ")
                file_message.text = text
                self.messages.append(file_message)
            else:
                user_message = Message(content=query)
                self.messages.append(user_message)

            response = self.client.work(messages=self.messages)
            self.messages.append(response)
            print()
            Utility.print2(f"BOT: {response.content}", color="blue")
            print()