from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='qwen/qwen3.8-27b')

messages = [
    SystemMessage(content="You are an AI assistant"),
    HumanMessage(content="Tell me about Jain Dharma")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)