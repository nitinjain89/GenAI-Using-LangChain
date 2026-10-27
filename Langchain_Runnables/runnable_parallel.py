from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='qwen/qwen3.8-27b',
                 max_tokens=300)

prompt1 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a tweet on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'linkedin': RunnableSequence(prompt1, model, parser),
    'tweet': RunnableSequence(prompt2, model, parser) 
})

result = chain.invoke({'topic': 'AI'})

print(result)