from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "Paris is the capital of France.",
    "France is a country in Europe.",
    "Berlin is the capital of Germany.",
    "The Eiffel Tower is located in Paris.",
    "Python is a popular programming language."
]

query_text = input("Your query: ")

query_vector = embedding.embed_query(query_text)
sentence_vectors = embedding.embed_documents(sentences)

scores = cosine_similarity([query_vector], sentence_vectors)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(sentences[index])
print('Similarity Socre is :', score)
