from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='qwen/qwen3.8-27b',
                 max_tokens=300)

prompt = PromptTemplate(
    template='write a summary for the following poem - {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('AI Poem.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))

# print(docs[0].page_content)
chain = prompt | model | parser

result = chain.invoke({'poem': docs[0].page_content})

print(result)