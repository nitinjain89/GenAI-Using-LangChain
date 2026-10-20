from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1",
#     task = "text-generation"
# )
# model = ChatHuggingFace(llm=llm)

model = ChatGroq(model='qwen/qwen3.8-27b')

result = model.invoke("What is the capital of France?")

print(result.content)