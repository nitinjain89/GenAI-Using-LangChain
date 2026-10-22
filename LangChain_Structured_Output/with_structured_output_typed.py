from langchain_groq import ChatGroq
from typing import TypedDict, Literal, Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='qwen/qwen3.8-27b')

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review"]
    summary: Annotated[str, "A bried summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either positive or negative"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I am very satisfied with the product quality and overall performance. The product works as expected and offers good value for money. However, the delivery was delayed by two days, and the packaging could have been better. Overall, it was a good purchasing experience, but there is room for improvement in delivery and packaging.""")

print(result)
print("Sentiment:", result['sentiment'])