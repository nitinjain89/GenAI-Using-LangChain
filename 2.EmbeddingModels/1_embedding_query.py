from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=100)

query_text = "What is the capital of France?"

query_vector = embeddings.embed_query(query_text)

print(len(query_vector))