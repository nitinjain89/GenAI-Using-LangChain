from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=10)

documents = [
    "Delhi is the capital of India",
    "Paris is the capital of France",
    "Jaipur is the capital of Rajasthan"
]

result = embeddings.embed_documents(documents)

print(result)