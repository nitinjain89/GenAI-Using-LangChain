from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='qwen/qwen3.8-27b',
                 max_tokens=300)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Answer the following question {question} using following text - {text}',
    input_variables=['question', 'text']
)

url = 'https://www.amazon.in/gp/product/B0DSKNKCYX?th=1&aref=VrDJYnGzEW&pd_rd_plhdr=t&pf_rd_r=483116SWG8G212V0RW38'

loader = WebBaseLoader(url)

docs = loader.load()

# print(docs[0].metadata)
chain = prompt | model | parser

while True:
    question = input("Question: ")
    if question == 'exit':
        break
    else:
        result = chain.invoke({'question':question,'text':docs[0].metadata})

    print(result)