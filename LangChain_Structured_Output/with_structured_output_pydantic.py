from langchain_groq import ChatGroq
from typing import TypedDict, Optional, Annotated, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='qwen/qwen3.8-27b')

# schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review")
    summary: str = Field(description="A bried summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description= "Return sentiment of the review either positive or negative")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cpns inside a list")
    name: Optional[str] = Field(description="Write the name of the reviewer")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I am very satisfied with the product quality and overall performance. The product works as expected and offers good value for money. However, the delivery was delayed by two days, and the packaging could have been better. Overall, it was a good purchasing experience, but there is room for improvement in delivery and packaging.""")

print(result)