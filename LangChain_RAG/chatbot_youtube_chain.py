from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs
import re

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

VIDEO_ID = "z_MoCiIfEPI"

def extract_video_id(video_url):
    parsed_url = urlparse(video_url.strip())
    hostname = parsed_url.hostname or ""

    if hostname in ["youtu.be", "www.youtu.be"]:
        video_id = parsed_url.path.lstrip("/").split("/")[0]

    elif hostname.endswith("youtube.com"):
        if parsed_url.path == "/watch":
            video_id = parse_qs(parsed_url.query).get("v", [""])[0]
        else:
            parts = parsed_url.path.strip("/").split("/")
            video_id = parts[1] if len(parts) >= 2 and parts[0] in [
                "shorts", "embed", "live"
            ] else ""

    else:
        raise ValueError("Please enter a valid YouTube URL.")

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Invalid YouTube video URL.")

    return video_id
    return hostname

def format_docs(retrieved_docs):
    """Combine retrieved document chunks into one context string."""
    context_text = "\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def build_chain(video_url):
    """Fetch transcript, build FAISS index, and return the RAG chain."""
    # 1. Extract video ID
    video_id = extract_video_id(video_url)

    # 2. Fetch transcript
    ytt_api = YouTubeTranscriptApi()

    transcript_data = ytt_api.fetch(
        video_id,
        languages=["en"]
    )

    transcript_items = transcript_data.to_raw_data()
    transcript = " ".join(item['text'] for item in transcript_items)

    if not transcript.strip():
        raise RuntimeError("The YouTube transcript is empty.")

    # 2. Split the transcripts into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    # 3. Initialize LLM and embedding model
    llm = ChatGroq(model='qwen/qwen3.8-27b', max_tokens=500)
    emb_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', output_dimensionality=1024)

    # 4. Create FAISS vector store and retriever
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=emb_model
    )

    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k':4}
    )

    # 5. Prompt template
    prompt = PromptTemplate(
        template="""You're a helpful assistant.
        Answer ONLY from the provided transcript text.
        If context is insufficient, just say you don't know. 
        
        Context:
        {context}
        Question: {question}""",
        input_variables=['context', 'question']
    )

    # 6. Build LCEL chain
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser

    return main_chain