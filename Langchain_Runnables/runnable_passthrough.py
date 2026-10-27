from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='qwen/qwen3.8-27b',
                 max_tokens=300)

prompt = PromptTemplate(
    template='Generate a joke on the {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explaine the following joke - {text}',
    input_variables=['text']
)

parser = StrOutputParser()

joke_gen_chain = prompt | model | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explaination': prompt2 | model | parser
})

final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({'topic': 'AI'})

print(result)