from langchain_ollama import ChatOllama

llm = ChatOllama(
    model='qwen3.5:0.8b'
)

response = llm.invoke("what is ollama?")
print(response.content)