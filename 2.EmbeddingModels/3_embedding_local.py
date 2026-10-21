from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

query_text = "What is the capital of France?"

vector = embedding.embed_query(query_text)

print(len(vector))