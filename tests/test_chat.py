from agentine import Message, LLM

msg = Message(
    role="user",
    content=[
        {
            "type": "text",
            "text": "Describe what is in this image:",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
                "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
                "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            },
        },
        {
            "type": "text",
            "text": "Then list the color found in the image.",
        },
    ],
)

llm = LLM(provider="openai")
result = llm.chat(messages=[msg])

print(result)