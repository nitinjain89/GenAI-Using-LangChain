from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """In circuits deep and data wide,  
A spark of thought begins to glide.  
Through lines of code and patterns bright,  
AI awakens in the night.

It learns from words, from art, from sound,  
Finds hidden truths that once lay bound.  
It answers questions, helps us see,  
A world of endless possibility.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0
)

result = splitter.split_text(text)

print(result)