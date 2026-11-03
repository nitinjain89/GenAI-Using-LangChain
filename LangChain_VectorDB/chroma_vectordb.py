from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

sentences = [
    "Python is a popular programming language.",
    "Python is widely used for machine learning.",
    "Machine learning helps computers learn from data.",
    "Deep learning uses neural networks.",
    "Neural networks are inspired by the human brain.",
    "LangChain helps developers build LLM applications.",
    "Large language models can understand human language.",
    "Vector databases store numerical representations of text.",
    "Chroma is a vector database used for semantic search.",
    "Semantic search finds text based on meaning.",
    "Embeddings convert text into numerical vectors.",
    "RAG retrieves relevant documents before generating an answer.",
    "Data science involves statistics and programming.",
    "SQL is used to query relational databases.",
    "Pandas is a Python library for data analysis."
]

documents = []
for i, sentence in enumerate(sentences):
    doc = Document(
        page_content=sentence,
        metadata={"id": i+1}
    )
    documents.append(doc)


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=100)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db",
    collection_name="sample"
)

vector_store.add_documents(documents)

print(vector_store.get(include=['metadatas']))