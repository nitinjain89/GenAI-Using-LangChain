from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())

model = ChatGroq(model='qwen/qwen3.8-27b',
                 max_tokens=300)

prompt = PromptTemplate(
    template='Generate a joke on the {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

gen_joke_chain = prompt | model | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = gen_joke_chain | parallel_chain

result = final_chain.invoke({'topic': 'AI'})

final_result = """Joke -\n{} \n\nword count - {}""".format(result['joke'], result['word_count'])

print(final_result)